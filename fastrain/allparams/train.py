import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import transformers
import trl
import torch
from transformers import HfArgumentParser, TrainingArguments, set_seed
from trl import SFTTrainer
from utils import create_and_prepare_model
import json
import mlflow
from transformers import DataCollatorForLanguageModeling
from datasets import Dataset, load_dataset, load_from_disk
import warnings
warnings.filterwarnings("ignore")

# parse args
@dataclass
class ModelArguments:
	
	model_name_or_path: str = field(
		metadata={"help": "Path to pretrained model or model identifier"}
		)
	lora_alpha: Optional[int] = field(default=16)
	lora_dropout: Optional[float] = field(default=0.)
	lora_r: Optional[int] = field(default=64)
	lora_target_modules: Optional[str] = field(
		default="q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj",
		)
	use_nested_quant: Optional[bool] = field(
		default=False
		)
	bnb_4bit_compute_dtype: Optional[str] = field(
		default="float16",
		metadata={"help": "Compute dtype for 4bit base model"}
		)
	bnb_4bit_quant_storage_dtype: Optional[str] = field(
		default="uint8"
		)
	bnb_4bit_quant_type: Optional[str] = field(
		default="nf4"
		)
	use_flash_attn: Optional[bool] = field(
		default=False
		)
	use_peft_lora: Optional[bool] = field(
		default=True
		)
	use_8bit_quantization: Optional[bool] = field(
		default=False
		)
	use_4bit_quantization: Optional[bool] = field(
		default=False
		)
	use_reentrant: Optional[bool] = field(
		default=False
		)


@dataclass
class DataTrainingArguments:
	dataset_path: Optional[str] = field(
		default=None
		)
	#packing: Optional[bool] = field(
	#	default=False
	#	)
	dataset_text_field: str = field(
		default="text",
		metadata={"help": "Dataset field to use as input text"}
		)
	max_seq_length: Optional[int] = field(default=512)
	append_concat_token: Optional[bool] = field(
		default=False,
		metadata={"help": "If True, appends `eos_token_id` at the end of each sample being packed."}
		)
	add_special_tokens: Optional[bool] = field(
		default=False,
		metadata={"help": "If True, tokenizer adds special tokens to each sample being packed"}
		)
	splits: Optional[str] = field(
		default="train,test"
		)

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


def main(model_args, data_args, training_args):
	set_seed(training_args.seed)
	model, peft_config, tokenizer = create_and_prepare_model(model_args, data_args, training_args)

	# gradient checkpoint utils
	model.config.use_cache = not training_args.gradient_checkpointing
	if training_args.gradient_checkpointing:
		training_args.gradient_checkpointing_kwargs = {"use_reentrant": model_args.use_reentrant}

	data_path = data_args.dataset_path

	if 'cots' in data_path:
		dataset = load_dataset("open-r1/codeforces-cots", "solutions_decontaminated", split="train")
	else:
		dataset = load_from_disk(data_path)
	print ('dataset loaded')
	print (len(dataset))
	block_text = len(dataset) == 1
	if block_text:
		train_data = tokenize_input(dataset, tokenizer, tile_size=data_args.max_seq_length)
		test_data = tokenize_input(dataset, tokenizer, tile_size=data_args.max_seq_length)
		train_text, test_text = detokenize_input(train_data, tokenizer), detokenize_input(test_data, tokenizer)
		# for sft trainer
		train_text = {'text': list(train_text)}
		train_text_dataset = Dataset.from_dict(train_text)

		test_text = {'text': list(test_text)}
		test_text_dataset = Dataset.from_dict(test_text)

	else:
		split_index=200
		train_text, test_text = dataset.skip(split_index), dataset.take(split_index)

	collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

	training_args.packing=False
	training_args.dataset_text_field=data_args.dataset_text_field
	#training_args.gradient_accumulation_steps = 8 # TODO: swap

	trainer = SFTTrainer(
		model=model,
		tokenizer=tokenizer,
		args=training_args,
		train_dataset=train_text,
		eval_dataset=test_text,
		peft_config=peft_config,
	)

	trainer.accelerator.print(f"{trainer.model}")

	# saving final model
	if trainer.is_fsdp_enabled:
	    trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")

	checkpoint=None
	if training_args.resume_from_checkpoint:
		checkpoint = training_args.resume_from_checkpoint
		print (f'Training initialized from checkpoint {checkpoint}')
	trainer.train(resume_from_checkpoint=checkpoint)
	trainer.save_model()


if __name__ == "__main__":
	parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
	model_args, data_args, training_args = parser.parse_args_into_dataclasses()
	main(model_args, data_args, training_args)
