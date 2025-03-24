PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
OMP_NUM_THREADS=8 accelerate launch --config_file "configs/fsdp_config_allparams.yaml" train.py \
--seed 100 \
--model_name_or_path "/home/bbadger/Desktop/llama-3.1-8b-instruct" \
--dataset_path "open-r1/codeforces-cots" \
--add_special_tokens False \
--append_concat_token False \
--max_seq_len 32768 \
--num_train_epochs 2 \
--logging_steps 10 \
--log_level "info" \
--logging_strategy "steps" \
--evaluation_strategy "steps" \
--eval_steps 50 \
--save_strategy "steps" \
--save_steps 500 \
--bf16 False \
--fp16 True \
--learning_rate 3e-5 \
--lr_scheduler_type "cosine" \
#--lr_scheduler_kwargs {min_lr_rate: 0.1} \
--weight_decay 0.0 \
--warmup_ratio 0.0 \
--max_grad_norm 1.0 \
--output_dir "/home/bbadger/experiments/llama-3.1-8b-codeforcescots" \
--per_device_train_batch_size 4 \
--per_device_eval_batch_size 4 \
--gradient_checkpointing True \
--gradient_accumulation_steps 1 \
--dataset_text_field "messages" \
--use_flash_attn False \
--use_peft_lora False \
--report_to "none" \
#--resume_from_checkpoint "/home/bbadger/experiments/llama-3.1-8b-codeforcescots/checkpoint-2000"
