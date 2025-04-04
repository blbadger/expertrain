PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
OMP_NUM_THREADS=8 accelerate launch --config_file "configs/fsdp_config_allparams.yaml" train.py \
--seed 100 \
--model_name_or_path "/home/badger/llama-3.1-8b-instruct" \
--dataset_path "open-r1/codeforces-cots" \
--add_special_tokens False \
--append_concat_token False \
--max_seq_len 32768 \
--num_train_epochs 3 \
--logging_steps 10 \
--log_level "info" \
--logging_strategy "steps" \
--evaluation_strategy "steps" \
--eval_steps 100 \
--save_strategy "steps" \
--save_steps 200 \
--bf16 True \
--fp16 False \
--learning_rate 4e-5 \
--lr_scheduler_type "cosine" \
--weight_decay 0.0 \
--warmup_ratio 0.0 \
--max_grad_norm 1.0 \
--output_dir "/home/badger/llama-3.1-8b-cfcots" \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 1 \
--gradient_checkpointing True \
--gradient_accumulation_steps 4 \
--dataset_text_field "messages" \
--use_flash_attn True \
--use_peft_lora False \
--report_to "none" \
