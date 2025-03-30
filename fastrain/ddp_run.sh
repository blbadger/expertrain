torchrun --nproc_per_node=4 train.py \
--seed 100 \
--model_name_or_path "/home/bbadger/Desktop/llama-3.1-8b-instruct" \
--dataset_path "open-r1/codeforces-cots" \
--add_special_tokens False \
--append_concat_token False \
--max_seq_len 1024 \
--num_train_epochs 5 \
--logging_steps 1 \
--log_level "info" \
--logging_strategy "steps" \
--evaluation_strategy "steps" \
--eval_steps 100 \
--save_strategy "steps" \
--save_steps 200 \
--bf16 False \
--fp16 True \
--packing False \
--learning_rate 4e-6 \
--lr_scheduler_type "cosine" \
--weight_decay 0.0 \
--warmup_ratio 0.0 \
--max_grad_norm 1.0 \
--output_dir "/home/bbadger/experiments/quen-cfcots-lora" \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 1 \
--gradient_checkpointing True \
--use_reentrant True \
--gradient_accumulation_steps 4 \
--dataset_text_field "content" \
--use_flash_attn False \
--use_peft_lora True \
--lora_r 64 \
--lora_alpha 64 \
--lora_dropout 0. \
--lora_target_modules "all-linear" \
--use_4bit_quantization True \
--use_nested_quant True \
--bnb_4bit_compute_dtype "float16" \
--bnb_4bit_quant_storage_dtype "float16" \
--report_to "none"
#--resume_from_checkpoint "/home/bbadger/experiments/llama-3.1-8b-codeforcescots-qlora/checkpoint-700"
