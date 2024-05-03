# Copyright 2021 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import argparse

os.environ['WANDB_DISABLED'] = 'true'
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["CUDA_DEVICE_ORDER"] ="PCI_BUS_ID"
os.environ['CUDA_LAUNCH_BLOCKING'] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from strgen import StringGenerator as SG

from transformers import T5TokenizerFast, T5ForConditionalGeneration,BitsAndBytesConfig

import argparse
import datetime
from pytz import timezone

#setup
import kss
import pandas as pd
import json
import logging
logging.disable(logging.INFO)
import os
import re
import sys
import random

import nltk  # Here to have a nice missing dependency error message early on
import numpy as np
from datasets import load_dataset, load_metric
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import log_softmax, softmax
import transformers
from filelock import FileLock
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    MBartTokenizer,
    MBartTokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import (
    get_last_checkpoint, 
    is_main_process, 
    default_compute_objective,
    # default_hp_space_ray
)


from utils import (
    default_linearize_table_context,
    sample_linearize_table_context,
    linearization_dic,
    save_json
)
from Args import ModelArguments, DataTrainingArguments, summarization_name_mapping

# https://www.kaggle.com/code/paultimothymooney/fine-tune-flan-t5-with-peft-lora-deeplearning-ai
from peft import PeftModel, LoraModel, PeftConfig, LoraConfig, get_peft_model, TaskType

with FileLock(".lock") as lock:
    nltk.download("punkt", quiet=True)

class DistillationTrainer(Seq2SeqTrainer):
    def __init__(self, *args, teacher_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher_model
        # place teacher on same device as student
        self._move_model_to_device(self.teacher,self.model.device)
        self.teacher.eval()

    def compute_loss(self, model, inputs, return_outputs=False):

        # compute student output
        outputs_student = model(**inputs)
        student_loss=outputs_student.loss
        # compute teacher output
        with torch.no_grad():
          outputs_teacher = self.teacher(**inputs)
        
        # assert size
        assert outputs_student.logits.size() == outputs_teacher.logits.size()

        vocab_size = outputs_student.logits.size(-1)
        ce_logits = outputs_student.logits.view(-1, vocab_size)
        ce_labels = inputs["labels"].view(-1)
        ce_loss = torch.nn.functional.cross_entropy(ce_logits, ce_labels)
        student_log_probs = log_softmax(outputs_student.logits.view(-1, vocab_size) / self.args.temperature, dim=-1)
        teacher_probs = softmax(outputs_teacher.logits.view(-1, vocab_size) / self.args.temperature, dim=-1)

        distill_loss = torch.nn.functional.kl_div(
            student_log_probs, teacher_probs, reduction="batchmean"
        )
        loss = (1 - self.args.alpha) * ce_loss + (
            self.args.alpha * self.args.temperature**2 / self.args.per_device_train_batch_size**2
        ) * distill_loss

        """
        # Soften probabilities and compute distillation loss
        loss_function = nn.KLDivLoss(reduction="batchmean")
        loss_logits = (loss_function(
            F.log_softmax(outputs_student.logits / self.args.temperature, dim=-1),
            F.softmax(outputs_teacher.logits / self.args.temperature, dim=-1)) * (self.args.temperature ** 2))
        # Return weighted student loss
        loss = self.args.alpha * student_loss + (1. - self.args.alpha) * loss_logits
        """

        return (loss, outputs_student) if return_outputs else loss

def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    return f"trainable model parameters: {trainable_model_params}\nall model parameters: {all_model_params}\npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%"
   

def postprocess_text(preds, labels, metric_name, tokenizer):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    if metric_name == "rouge":

        preds = [pred.split(' ') for pred in preds]
        labels = [[label.split(' ')] for label in labels]
        """
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
        
        
        preds =  ["\n".join(kss.split_sentences(pred, use_heuristic=False,
                                        num_workers=32)) for pred in preds]
        
        labels =  ["\n".join(kss.split_sentences(label, use_heuristic=False,
                                        num_workers=32)) for label in labels]
        """
    elif metric_name == "sacrebleu":  # sacrebleu
        labels = [[label] for label in labels]
    elif metric_name == "bleu":
        preds = [pred.split(' ') for pred in preds]
        labels = [[label.split(' ')] for label in labels]
    else:
        pass

    return preds, labels


def main(model_args, data_args, training_args, output_dir, teacher_model_args, teacher_data_args, teacher_training_args):
    

    # Detecting last checkpoint.
    last_checkpoint = None
    if not model_args.hyper_param_search:
        
        if os.path.isdir(output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
            last_checkpoint = get_last_checkpoint(output_dir)
            if last_checkpoint is None and len(os.listdir(output_dir)) > 0:
                raise ValueError(
                    f"Output directory ({output_dir}) already exists and is not empty. "
                    "Use --overwrite_output_dir to overcome."
                )
            elif last_checkpoint is not None:
                logger.info(
                    f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                    "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
                )
    if model_args.distillation == 'true':
        teacher_config = AutoConfig.from_pretrained(
            teacher_model_args.config_name if teacher_model_args.config_name else model_args.teacher_model_name_or_path,
            cache_dir=teacher_model_args.cache_dir,
            revision=teacher_model_args.model_revision,
            use_auth_token=True if teacher_model_args.use_auth_token else None,
        )

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    tokenizer = T5TokenizerFast.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    """ 
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    """ 
    if data_args.linearization_strategy != "concat":
        tokenizer.add_special_tokens({'cls_token':'[CLS]','sep_token':'[SEP]'})

    model = None
    if model_args.hyper_param_search:
        def model_init():
            # model = AutoModelForSeq2SeqLM.from_pretrained(
            model = T5ForConditionalGeneration.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
            )
            
            model.resize_token_embeddings(len(tokenizer))

            return model
    else:
        if model_args.lora_adapter == 'false':
            # model = AutoModelForSeq2SeqLM.from_pretrained(
            model = T5ForConditionalGeneration.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
            )

        if model_args.lora_adapter == 'true':

            bnb_config = BitsAndBytesConfig(
                # load_in_8bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )

            model = T5ForConditionalGeneration.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
                quantization_config=bnb_config,
            )

            lora_config = LoraConfig(
                r=model_args.lora_rank, # Rank
                lora_alpha=model_args.lora_alpha, # Scaling Factor
                target_modules=["q", "v"], # Query & Value
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.SEQ_2_SEQ_LM # FLAN-T5
            )
                
            model = get_peft_model(model, lora_config) 
            print("\n")
            print(print_number_of_trainable_model_parameters(model), "\n")

        if data_args.linearization_strategy != 'concat':
            special_tokens_dict = {'additional_special_tokens': ['[CEL]', '[C1]','[C2]','[C3]','[C4]']}
            num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
            model.resize_token_embeddings(len(tokenizer))
        if model_args.model_name_or_path == "t5-large":
            print('CUDA device count:', torch.cuda.device_count())
            # Uncomment to allow naive model parallelization!
#             if torch.cuda.device_count() == 8:
#                 device_map = {0: [0, 1],
#                         1: [2, 3, 4],
#                         2: [5,6,7],
#                         3: [8,9,10],
#                         4:[11,12,13],
#                         5:[14,15,16],
#                         6:[17,18,19],
#                         7:[20,21,22,23]
#                     }
#             else:
#                 device_map = {0: [0, 1, 2],
#                     1: [3, 4, 5, 6, 7, 8, 9],
#                     2: [10, 11, 12, 13, 14, 15, 16],
#                     3: [17, 18, 19, 20, 21, 22, 23]}

#             model.parallelize(device_map)



    # load dataset
    data_files = {}
    if data_args.train_file is not None:
        data_files["train"] = data_args.train_file
        extension = data_args.train_file.split(".")[-1]
    if data_args.validation_file is not None:
        data_files["validation"] = data_args.validation_file
        extension = data_args.validation_file.split(".")[-1]
    if data_args.test_file is not None:
        data_files["test"] = data_args.test_file
        extension = data_args.test_file.split(".")[-1]

    datasets = load_dataset(extension, data_files=data_files, field='data')

    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if training_args.fp16 else None,
        )
    
    if model_args.hyper_param_search:
        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            if isinstance(preds, tuple):
                preds = preds[0]
            decoded_raw_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
            if data_args.ignore_pad_token_for_loss:
                # Replace -100 in the labels as we can't decode them.
                labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            decoded_raw_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
            result = {}
            metric = load_metric('sacrebleu')
            decoded_preds, decoded_labels = postprocess_text(decoded_raw_preds, decoded_raw_labels, 'sacrebleu', tokenizer)
            res = metric.compute(predictions=decoded_preds, references=decoded_labels)
            result['sacrebleu'] = res["score"]
            return result

    else:
        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            if isinstance(preds, tuple):
                preds = preds[0]
  
            decoded_raw_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
            if data_args.ignore_pad_token_for_loss:
                # Replace -100 in the labels as we can't decode them.
                labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            decoded_raw_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            decoded_raw_preds = [' '.join(pred.split()) for pred in decoded_raw_preds]

            prediction_df = pd.DataFrame({'label':decoded_raw_labels, 
                                          'prediction':decoded_raw_preds})

            
            dic_pred_label = {'predictions': decoded_raw_preds, 'labels': decoded_raw_labels}
            save_json(dic_pred_label, os.path.join(output_dir, "detokenized_outputs.json"))

            result = {}
            # Some simple post-processing
            # for metric_name in [data_args.metric_names]:
            for metric_name in ["bleu","sacrebleu", "rouge", "meteor", "bertscore", "bleurt"]:
                metric = load_metric(metric_name)
                decoded_preds, decoded_labels = postprocess_text(decoded_raw_preds, decoded_raw_labels, metric_name, tokenizer)

                prediction_df2 = pd.DataFrame({'label':decoded_raw_labels, 
                                                'prediction':decoded_raw_preds})
                
                if metric_name == "bertscore":
                    res = metric.compute(predictions=decoded_preds, references=decoded_labels, lang="ko")
                    for k,v in res.items():
                        if k =="hashcode":
                            continue
                        result[f"{metric_name}_{k}_0"] = round(v[0], 2)
                        result[f"{metric_name}_{k}_1"] = round(v[1], 2)
                else:
                    res = metric.compute(predictions=decoded_preds, references=decoded_labels)

                    if metric_name == "sacrebleu":
                        result[metric_name] = res["score"]
                    elif metric_name == "bleurt":
                        result[f"{metric_name}_0"] = round(res["scores"][0], 2) 
                        result[f"{metric_name}_1"] = round(res["scores"][1], 2) 
                    elif metric_name == "rouge":
                        res = metric.compute(predictions=decoded_preds,
                                                references=[[l] for l in decoded_labels],
                                                use_stemmer=True)
                        res = {key: value.mid.fmeasure * 100 for key, value in res.items()}
                        for k, v in res.items():
                            result[f"{metric.name}_{k}"] = np.mean(v)
                    else:
                        result[metric_name] = res[metric_name]

            prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
            result["gen_len"] = np.mean(prediction_lens)
            result = {k: round(v, 4) for k, v in result.items()}

            evaluation_log_path = output_dir + "/evaluation_log" + ".csv"

            if os.path.isfile(evaluation_log_path) == True:
                evaluation_log_df = pd.read_csv(evaluation_log_path) 
                
            elif os.path.isfile(evaluation_log_path) == False: 
                evaluation_log_df = pd.DataFrame({'Epoch':[0], 'Bleu':[0], 'Sacrebleu':[0], 
                                                  'Rouge 1':[0], 'Rouge 2':[0], 'Rouge L':[0], 'Rouge Lsum':[0],
                                                  'Meteor':[0], 
                                                  'Bertscore Precison 0':[0], 'Bertscore Precison 1':[0], 
                                                  'Bertscore Recall 0':[0], 'Bertscore Recall 1':[0], 
                                                  'Bertscore F1 0':[0], 'Bertscore F1 1':[0], 
                                                  'Bleurt 0':[0], 'Bleurt 1':[0], 
                                                  'Generated Length':[0]})

            save_csv_file = output_dir + "/detokenized_outputs" + "_" + str(len(evaluation_log_df)) + ".csv"
            prediction_df.to_csv(save_csv_file, index=False)
            # save_csv_file = save_csv_file.replace(".csv", "_000.csv")
            # prediction_df2.to_csv(save_csv_file, index=False)

            evaluation_log_df.loc[len(evaluation_log_df)] = [len(evaluation_log_df),
                                                               result["bleu"], result["sacrebleu"],
                                                               result["rouge_rouge1"], 
                                                               result["rouge_rouge2"], 
                                                               result["rouge_rougeL"], 
                                                               result["rouge_rougeLsum"],
                                                               result["meteor"],
                                                               result["bertscore_precision_0"], 
                                                               result["bertscore_precision_1"],
                                                               result["bertscore_recall_0"],
                                                               result["bertscore_recall_1"],
                                                               result["bertscore_f1_0"],
                                                               result["bertscore_f1_1"],
                                                               result["bleurt_0"],
                                                               result["bleurt_1"],                                                               
                                                               result["gen_len"]]
            
            evaluation_log_df = evaluation_log_df.astype({'Epoch':'int'})
            evaluation_log_df.to_csv(evaluation_log_path, index=False)

            print()
            print("Epoch : ", str(len(evaluation_log_df)-1))
            print()

            return result

    linearize_method = [linearization_dic[i] for i in data_args.linearization_strategy]
    # TODO: move linearize method to data_args
    train_dataset, eval_dataset, test_dataset = preprocess(
        model_args, 
        data_args, 
        training_args, 
        datasets, 
        tokenizer, 
        linearize_method
        )

    if train_dataset is None and eval_dataset is None and test_dataset is None:
        return
    best_run = None
    if model_args.hyper_param_search and training_args.do_train:
        trainer = Seq2SeqTrainer(
            model_init=model_init,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics if training_args.predict_with_generate else None,

        )

        def my_hp_space_ray(trial):
            from ray import tune

            return {
                "learning_rate": tune.loguniform(1e-6, 1e-4),
                "num_train_epochs": tune.choice(list(range(6, 30))),
                "seed": tune.choice(list(range(1, 41))),
                "per_device_train_batch_size": tune.choice([4, 8, 16]),
            }

        best_run = trainer.hyperparameter_search(
            n_trials=10, 
            direction="maximize", 
            checkpoint_freq = 500,
            compute_objective = default_compute_objective,
            hp_space = my_hp_space_ray
        )
        save_json(best_run.hyperparameters,os.path.join(output_dir, "best_run.json"))
        for n, v in best_run.hyperparameters.items():
            setattr(trainer.args, n, v)
        exit(0)

    else:
        
        if model is None:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
            )
            model.resize_token_embeddings(len(tokenizer))

        if model_args.distillation == 'true':

            teacher_model = AutoModelForSeq2SeqLM.from_pretrained(
                model_args.teacher_model_name_or_path,
                from_tf=bool(".ckpt" in model_args.teacher_model_name_or_path),
                config=teacher_config,
                cache_dir=teacher_model_args.cache_dir,
                revision=teacher_model_args.model_revision,
                use_auth_token=True if teacher_model_args.use_auth_token else None,
            )
            model.resize_token_embeddings(len(tokenizer))

            trainer = DistillationTrainer(
                model,
                training_args,
                teacher_model=teacher_model,
                train_dataset=train_dataset if training_args.do_train else None,
                eval_dataset=eval_dataset if training_args.do_eval else None,
                tokenizer=tokenizer,
                data_collator=data_collator,
                compute_metrics=compute_metrics # if training_args.predict_with_generate else None,
            )

        if model_args.distillation == 'false':
            trainer = Seq2SeqTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset if training_args.do_train else None,
                eval_dataset=eval_dataset if training_args.do_eval else None,
                tokenizer=tokenizer,
                data_collator=data_collator,
                compute_metrics=compute_metrics # if training_args.predict_with_generate else None,
            )
       
    #------ Main Meat -------
    all_metrics = {}
    # Training
    if training_args.do_train:
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        elif os.path.isdir(model_args.model_name_or_path):
            checkpoint = model_args.model_name_or_path
        else:
            checkpoint = None
        # print("\n\n")
        # print(checkpoint)
        train_result = trainer.train()
        # train_result = trainer.train(resume_from_checkpoint=checkpoint) if best_run is None else trainer.train()
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))
        if trainer.is_world_process_zero():
            metrics_formatted = trainer.metrics_format(metrics)
            
            logger.info("***** train metrics *****")
            k_width = max(len(str(x)) for x in metrics_formatted.keys())
            v_width = max(len(str(x)) for x in metrics_formatted.values())
            for key in sorted(metrics_formatted.keys()):
                logger.info(f"  {key: <{k_width}} = {metrics_formatted[key]:>{v_width}}")

            save_json(metrics, os.path.join(output_dir, "train_results.json"))
            all_metrics.update(metrics)

            # Need to save the state, since Trainer.save_model saves only the tokenizer with the model
            trainer.state.save_to_json(os.path.join(output_dir, "trainer_state.json"))

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate(
            max_length=data_args.val_max_target_length, num_beams=data_args.num_beams, metric_key_prefix="eval"
        )
        max_val_samples = data_args.max_val_samples if data_args.max_val_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_val_samples, len(eval_dataset))

        if trainer.is_world_process_zero():
            metrics_formatted = trainer.metrics_format(metrics)

            logger.info("***** val metrics *****")
            k_width = max(len(str(x)) for x in metrics_formatted.keys())
            v_width = max(len(str(x)) for x in metrics_formatted.values())
            for key in sorted(metrics_formatted.keys()):
                logger.info(f"  {key: <{k_width}} = {metrics_formatted[key]:>{v_width}}")

            random_number = str(random.randint(0, 10))
            save_json(metrics, os.path.join(output_dir, random_number + "_" + "eval_results.json"))
            all_metrics.update(metrics)

    if training_args.do_predict:
        logger.info("*** Test ***")

        test_results = trainer.predict(
            test_dataset,
            metric_key_prefix="test",
            max_length=data_args.val_max_target_length,
            num_beams=data_args.num_beams,
        )
        metrics = test_results.metrics
        max_test_samples = data_args.max_test_samples if data_args.max_test_samples is not None else len(test_dataset)
        metrics["test_samples"] = min(max_test_samples, len(test_dataset))

        if trainer.is_world_process_zero():
            metrics_formatted = trainer.metrics_format(metrics)
            
            logger.info("***** test metrics *****")
            k_width = max(len(str(x)) for x in metrics_formatted.keys())
            v_width = max(len(str(x)) for x in metrics_formatted.values())
            for key in sorted(metrics_formatted.keys()):
                logger.info(f"  {key: <{k_width}} = {metrics_formatted[key]:>{v_width}}")

            random_number = str(random.randint(0, 10))
            save_json(metrics, os.path.join(output_dir, random_number + "_" + "test_results.json"))
            all_metrics.update(metrics)

            if training_args.predict_with_generate:
                test_preds = tokenizer.batch_decode(
                    test_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                test_preds = [pred.strip() for pred in test_preds]
    
                output_test_preds_file = os.path.join(output_dir, "test_preds_seq2seq.txt")
                with open(output_test_preds_file, "w") as writer:
                    writer.write("\n".join(test_preds))

    


#TODO: make a 'run' class  
def preprocess(model_args, data_args, training_args, datasets,tokenizer, linearize_method):
    #----Preprocess datasets here
    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    if training_args.do_train:
        column_names = datasets["train"].column_names
    elif training_args.do_eval:
        column_names = datasets["validation"].column_names
    elif training_args.do_predict:
        column_names = datasets["test"].column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return
    dataset_columns = summarization_name_mapping.get(data_args.dataset_name, None)

    if data_args.text_column is None:
        text_column = dataset_columns[0] if dataset_columns is not None else column_names[6]
    else:
        text_column = data_args.text_column
    if data_args.summary_column is None:
        summary_column = dataset_columns[1] if dataset_columns is not None else column_names[0]
    else:
        summary_column = data_args.summary_column

    if data_args.context_column is None:
        context_column = dataset_columns[1] if dataset_columns is not None else column_names[4]
    else:
        context_column = data_args.context_column

    
    def preprocess_function(examples):
        # Temporarily set max_target_length for training.
        max_target_length = data_args.max_target_length
        padding = "max_length" if data_args.pad_to_max_length else False
        prefix = data_args.source_prefix if data_args.source_prefix is not None else "summarize: "
        if data_args.task.startswith("translation"):
            inputs = [ex[source_lang] for ex in examples["translation"]]
            targets = [ex[target_lang] for ex in examples["translation"]]
        else:
  
            """
            inputs = [
                linearize_method(table, question) for table,question in zip(examples[text_column],examples[context_column])
            ]
            targets = examples[summary_column]
            """

            inputs = [
                i_method(table, question) for i_method in linearize_method for table,question in zip(examples[text_column],examples[context_column])
            ]

            targets = [
                x for x in examples[summary_column] for y in range(len(linearize_method))
            ]
 
        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        # if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
        #     logger.warn(
        #         "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
        #         f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
        #     )
        return model_inputs

    train_dataset = None
    eval_dataset = None
    test_dataset = None
    if training_args.do_train:
        train_dataset = datasets["train"]
        if "train" not in datasets:
            raise ValueError("--do_train requires a train dataset")
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
        train_dataset = train_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )


    if training_args.do_eval:
        max_target_length = data_args.val_max_target_length
        if "validation" not in datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = datasets["validation"]
        if data_args.max_val_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_val_samples))
        eval_dataset = eval_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )


    if training_args.do_predict:
        max_target_length = data_args.val_max_target_length
        if "test" not in datasets:
            raise ValueError("--do_predict requires a test dataset")
        test_dataset = datasets["test"]
        if data_args.max_test_samples is not None:
            test_dataset = test_dataset.select(range(data_args.max_test_samples))
        test_dataset = test_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )
    return train_dataset, eval_dataset, test_dataset

if __name__ == "__main__":


    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))

    """
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, training_args = parser.parse_args_into_dataclasses()
    """

    parser_for_config = argparse.ArgumentParser()
    parser_for_config.add_argument("--config_name", type=str) 
    parser_for_config.add_argument("--topic", type=str) 
    parser_for_config.add_argument("--lora_adapter", type=str) 
    parser_for_config.add_argument("--cuda_visible_devices", type=int) 
    parser_for_config.add_argument("--lora_rank", type=int) 
    parser_for_config.add_argument("--lora_alpha", type=int)    
    parser_for_config.add_argument("--distillation", type=str) 
    parser_for_config.add_argument("--teacher_model_name_or_path", type=str)
    parser_for_config.add_argument("--teacher_config_name", type=str)
    parser_for_config.add_argument("--temperature", type=int)
    parser_for_config.add_argument("--alpha", type=float)
    args_for_config = parser_for_config.parse_args()

    model_args, data_args, training_args = parser.parse_json_file(json_file=args_for_config.config_name)
    teacher_model_args, teacher_data_args, teacher_training_args = parser.parse_json_file(json_file=args_for_config.teacher_config_name)
    model_args.distillation = args_for_config.distillation
    model_args.teacher_model_name_or_path = args_for_config.teacher_model_name_or_path
    training_args.temperature = args_for_config.temperature
    training_args.alpha = args_for_config.alpha

    model_args.lora_adapter = args_for_config.lora_adapter
    model_args.lora_rank = args_for_config.lora_rank
    model_args.lora_alpha = args_for_config.lora_alpha

    # Setup logging
    logger = logging.getLogger(__name__)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)
    
    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()

    file_paths = ["data_args.txt", "model_args.txt", "training_args.txt"]
    for file_path in file_paths:
        with open(file_path, "w") as f:
            if "data_args" in file_path:
                f.write(str(data_args))
            elif "model_args" in file_path:
                f.write(str(model_args))
            elif "training_args" in file_path:
                f.write(str(training_args))

    # logger.info("Data Parameters %s", data_args)
    # logger.info("Model Parameters %s", model_args)
    # logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    if 'small' in args_for_config.config_name: model_type = 'small'
    elif 'base' in args_for_config.config_name: model_type = 'base'
    elif 'large' in args_for_config.config_name: model_type = 'large'

    random_string = SG(r"[\w]{5}").render()
    now_time = str(datetime.datetime.now(timezone('Asia/Seoul')).strftime('%m-%d %H:%M'))
    output_dir = "{}/{} ({}-all) ({}) ({}) (revised)".format(training_args.output_dir, random_string, model_type, args_for_config.topic, now_time)
    training_args.output_dir = output_dir
    print()
    print("Output Folder: ", output_dir)
    print()

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)


    main(model_args,data_args, training_args, output_dir, teacher_model_args, teacher_data_args, teacher_training_args)
