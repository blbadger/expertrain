import unsloth
import os
from enum import Enum
import torch
from datasets import DatasetDict, load_dataset, load_from_disk
from datasets.builder import DatasetGenerationError
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig


def prepare_unsloth(args, data_args, training_args, device=None):
	from unsloth import FastModel, FastLanguageModel
	model, tokenizer = FastModel.from_pretrained(
	    model_name = args.model_name_or_path,
	    max_seq_length = data_args.max_seq_length, # Choose any for long context!
	    load_in_4bit = True,  # 4 bit quantization to reduce memory
	    load_in_8bit = False, # [NEW!] A bit more accurate, uses 2x memory
	    full_finetuning = False, # [NEW!] We have full finetuning now!
	)
	# Do model patching and add fast LoRA weights
	model = FastLanguageModel.get_peft_model(
	    model,
	    r = args.lora_r,
	    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
	                      "gate_proj", "up_proj", "down_proj",],
	    lora_alpha = args.lora_alpha,
	    lora_dropout = 0, # Supports any, but = 0 is optimized
	    bias = "none",    # Supports any, but = "none" is optimized
	    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
	    random_state = 3407,
	    max_seq_length = data_args.max_seq_length,
	    use_rslora = False,  # We support rank stabilized LoRA
	    loftq_config = None, # And LoftQ
	)
	peft_config = []
	return model, peft_config, tokenizer


def create_and_prepare_model(args, data_args, training_args, device=None):
	bnb_config = None
	quant_storage_dtype = None

	if args.use_4bit_quantization:
		compute_dtype = getattr(torch, args.bnb_4bit_compute_dtype)
		quant_storage_dtype = getattr(torch, args.bnb_4bit_quant_storage_dtype)

		bnb_config = BitsAndBytesConfig(
			load_in_4bit=args.use_4bit_quantization,
			bnb_4bit_quant_type = args.bnb_4bit_quant_type,
			bnb_4bit_compute_dtype = compute_dtype,
			bnb_4bit_use_double_quant = args.use_nested_quant,
			bnb_4bit_quant_storage = quant_storage_dtype
			)

		if args.use_8bit_quantization:
			bnb_config = BitsAndBytesConfig(load_in_8bit=True)

	torch_dtype = quant_storage_dtype if quant_storage_dtype and quant_storage_dtype.is_floating_point else torch.float32
	if not device:
		model = AutoModelForCausalLM.from_pretrained(
				args.model_name_or_path,
				quantization_config = bnb_config,
				trust_remote_code=True,
				attn_implementation="flash_attention_2" if args.use_flash_attn else "eager",
				torch_dtype=torch_dtype
			)
	else:
		model = AutoModelForCausalLM.from_pretrained(
				args.model_name_or_path,
				quantization_config = bnb_config,
				trust_remote_code=True,
				attn_implementation="flash_attention_2" if args.use_flash_attn else "eager",
				torch_dtype=torch_dtype,
				device_map=device
			)

	peft_config=None
	if args.use_peft_lora:
		peft_config = LoraConfig(
			lora_alpha=args.lora_alpha,
			lora_dropout=args.lora_dropout,
			r=args.lora_r,
			bias="none",
			task_type="CAUSAL_LM",
			target_modules=args.lora_target_modules.split(",")
			if args.lora_target_modules != "all-linear"
			else args.lora_target_modules
			)

	# TODO: add special token compatibility
	special_tokens=None
	if special_tokens is not None:
		tokenizer = AutoTokenizer.from_pretrained(
			args.model_name_or_path,
			pad_token=special_tokens.pad_token.value,
			bos_token=special_tokens.bos_token.value,
			eos_token=special_tokens.eos_token.value,
			additional_special_tokens=special_tokens.list(),
			trust_remote_code=True
			)
	else:
		tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
		tokenizer.pad_token = tokenizer.eos_token

	return model, peft_config, tokenizer


def tile_inputs(input_ids, tokenizer, tile_overlap=20, tile_size=1024):
	text_length = len(input_ids[0])
	assert text_length >= tile_overlap, "Text must be longer than overlap to tile"

	i, tiled_arr = 0, []
	while i < text_length:
		if i + tile_size <= text_length:
			tokens = input_ids[0][i:i+tile_size]
			tiled_arr.append(tokens)
		else:
			# pad the last tile
			tokens = input_ids[0][i:i+tile_size]
			pad_length = tile_size - len(tokens)
			tokens = torch.nn.functional.pad(
				tokens,
				(0, pad_length),
				mode='constant', 
				value=tokenizer.pad_token_id
				)
			tiled_arr.append(tokens)
		i += tile_size - tile_overlap
	return tiled_arr

def tokenize_input(text, tokenizer, tile_size=1024, overlap_size=20):
	# assumes dataset is not large (< 10b samples) and can be loaded in memory
	all_data = []
	for i, text_file in enumerate(text):
		if isinstance(text_file, dict):
			text_file = text_file['text']
		input_ids = tokenizer.encode(
			text_file,
			add_special_tokens=False,
			return_tensors="pt",
			truncation=False
			)

		if len(input_ids[0]) < overlap_size:
			continue
		data = tile_inputs(input_ids, tokenizer, tile_size=tile_size, tile_overlap=overlap_size)
		all_data += data
	return all_data


def detokenize_input(tokens, tokenizer):
	text = []
	for i, tensor in enumerate(tokens):
		text_input = tokenizer.decode(tensor, skip_special_tokens=True)
		text.append(text_input)
	return text
