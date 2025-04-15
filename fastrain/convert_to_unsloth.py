from unsloth import FastLanguageModel

max_seq_length = 2048 # Can increase for longer reasoning traces
lora_rank = 32 # Larger rank = smarter, but slower

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "/home/bbadger/experiments/qwen-coderinstruct-bird-8192/checkpoint-589/merged_model",
    max_seq_length = max_seq_length,
    load_in_4bit = True, # False for LoRA 16bit
    fast_inference = True, # Enable vLLM fast inference
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.55, # Reduce if out of memory
)

model.save_pretrained_merged("/home/bbadger/experiments/qwen-coderinstruct-bird-8192/checkpoint-589/merged_16bit_model", tokenizer, save_method = "merged_16bit",)
