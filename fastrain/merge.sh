python merge_adapters.py \
--model_name_or_path "/home/bbadger/Desktop/llama-3.1-8b-instruct" \
--lora_weights_path "/home/bbadger/experiments/llama-3.1-8b-codeforcescots-qlora-b128"
--use_peft_lora True \
--lora_r 64 \
--lora_alpha 64 \
--lora_dropout 0. \
--lora_target_modules "all-linear" \
--use_4bit_quantization False \
