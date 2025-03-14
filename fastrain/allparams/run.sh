OMP_NUM_THREADS=8 accelerate launch --config_file "configs/fsdp_config_allparams.yaml" train.py \
--seed 100 \
--model_name_or_path "/home/bbadger/Desktop/llama-3.1-8b-instruct" \
--dataset_path "/home/bbadger/experiments/github_pages_source" \
--add_special_tokens False \
--append_concat_token False \
--max_seq_len 16384 \
--num_train_epochs 11 \
--logging_steps 50 \
--log_level "info" \
--logging_strategy "steps" \
--evaluation_strategy "steps" \
--eval_steps 10 \
--save_strategy "steps" \
--save_steps 100 \
--bf16 False \
--fp16 True \
--learning_rate 2e-5 \
--lr_scheduler_type "linear" \
--weight_decay 0.0 \
--warmup_ratio 0.0 \
--max_grad_norm 1.0 \
--output_dir "/home/bbadger/experiments/github_full_llama3.1_8b" \
--per_device_train_batch_size 4 \
--per_device_eval_batch_size 2 \
--gradient_checkpointing True \
--dataset_text_field "content" \
--use_flash_attn False \
--use_peft_lora False \
--report_to "none"
