python -m train \
    --config_name "config/google-t5-small.json" \
    --topic "kd-input-512-output-64" \
    --lora_adapter 'false' \
    --lora_rank 256 \
    --lora_alpha 512 \
    --distillation 'false' \
    --teacher_model_name_or_path 'model/teacher-model/t5-base' \
    --teacher_config_name "config/google-t5-base.json" \
    --temperature 1 \
    --alpha 0.5

# --config_name "config/google-t5-base.json" \
# --config_name "config/google-t5-large.json" \
# --config_name "config/google-t5-base.json" \
# --config_name "config/google-t5-large.json" \
# --config_name "config/google-t5-base.json" \
