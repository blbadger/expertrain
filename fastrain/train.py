import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import transformers
import trl
import torch
from transformers import HfArgumentParser, TrainingArguments, set_seed
from trl import SFTTrainer, SFTConfig
from utils import create_and_prepare_model, tokenize_input, tile_inputs, detokenize_input
import json
import mlflow
from transformers import DataCollatorForLanguageModeling
from trl import DataCollatorForCompletionOnlyLM
from datasets import Dataset, load_dataset, load_from_disk, concatenate_datasets
import warnings
warnings.filterwarnings("ignore")


from datasets import disable_caching
disable_caching()

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
	packing: Optional[bool] = field(
		default=False
		)
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


def main(model_args, data_args, training_args):
	set_seed(training_args.seed)
	model, peft_config, tokenizer = create_and_prepare_model(model_args, data_args, training_args)

	# gradient checkpoint utils
	model.config.use_cache = not training_args.gradient_checkpointing
	if training_args.gradient_checkpointing:
		training_args.gradient_checkpointing_kwargs = {"use_reentrant": model_args.use_reentrant}

	data_path = data_args.dataset_path
	if 'cots' in data_path:
		dataset = load_dataset(data_path, "solutions_decontaminated", split="train", columns=["messages"])
		# python_dataset = load_dataset(data_path, "solutions_py_decontaminated", split="train")
		# dataset = concatenate_datasets((dataset, python_dataset))
	else:
		if 'huggingface' in data_path.lower():
			dataset = load_dataset(data_path, split="train", name="sample-10BT", streaming=False)
		elif os.path.exists(data_path):
			dataset = load_from_disk(data_path)
			print (dataset[0])
		else:
			print ("no dataset found")

	print ('dataset loaded')

	block_text = len(dataset) == 1
	print (f"Block text: {block_text}")
	if block_text:
		data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
		train_data = tokenize_input(dataset, tokenizer, tile_size=data_args.max_seq_len)
		dataset_split = int(len(train_data) * 0.8)
		train_data, test_data = train_data[:dataset_split], train_data[dataset_split:]
		train_text, test_text = detokenize_input(train_data, tokenizer), detokenize_input(test_data, tokenizer)

		# for sft trainer
		train_text = {'text': list(train_text)}
		test_text = {'text': list(test_text)}

	else:
		mock = [
			{"role": "user", "content":"@|@"},
			{"role": "assistant", "content":"@|@"},
		]
		instruction_template = tokenizer.decode(tokenizer.apply_chat_template(mock)).split("@|@")[0]		
		response_template = tokenizer.decode(tokenizer.apply_chat_template(mock)).split("@|@")[1]
		print (f"Input template: {instruction_template}Response template: {response_template}")
		data_collator = DataCollatorForCompletionOnlyLM(
			instruction_template=instruction_template,
			response_template=response_template,
			tokenizer=tokenizer, 
			mlm=False
		)
		if 'bird' in str(data_path):
			train_text = dataset
			test_text = load_from_disk('/home/bbadger/experiments/bird_dev_dataset')
		else:
			split_index=200
			train_text, test_text = dataset.skip(split_index), dataset.take(split_index)

	#todo: 8-bit optims fail to send params from cpu during the backward, see if this can be debugged
	training_args.max_length = data_args.max_seq_length
	training_args.max_seq_length = data_args.max_seq_length
	training_args.optim = "adamw_torch"	
	config = SFTConfig(
		**training_args.to_dict(),
		max_length = data_args.max_seq_length,
		max_seq_length = data_args.max_seq_length,	
	)

	# add training_args to SFTConfig if not sending training args to dict
	# for key in training_args.__dict__.keys():
	# 	if key in config.__dict__.keys():
	# 		config.__dict__[key] = training_args.__dict__[key]

	trainer = SFTTrainer(
		model=model,
		args=config,
		train_dataset=train_text,
		eval_dataset=test_text,
		peft_config=peft_config,
		data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
	)
	trainer.accelerator.print(f"{trainer.model}")
	trainer.model.print_trainable_parameters()

	# saving final model
#	if trainer.is_fsdp_enabled:
#	    trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")

	checkpoint=None
	if training_args.resume_from_checkpoint:
		checkpoint = training_args.resume_from_checkpoint
		print (f'Training initialized from checkpoint {checkpoint}')

	trainer.train(resume_from_checkpoint=checkpoint)
	
	if trainer.is_fsdp_enabled:
		trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")

	trainer.save_model()


if __name__ == "__main__":
	parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
	model_args, data_args, training_args = parser.parse_args_into_dataclasses()
	main(model_args, data_args, training_args)
