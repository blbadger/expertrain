import os
from enum import Enum
import torch
from datasets import DatasetDict, load_dataset, load_from_disk
from datasets.builder import DatasetGenerationError
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig

def create_and_prepare_model(args, data_args, training_args):
	peft_config = None
	torch_dtype = torch.float16
	model = AutoModelForCausalLM.from_pretrained(
			args.model_name_or_path,
			trust_remote_code=True,
			attn_implementation="flash_attention_2" if args.use_flash_attn else "eager",
			torch_dtype=torch_dtype
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
	# assumes dataset is not large (< 1b samples) and can be loaded in memory
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

