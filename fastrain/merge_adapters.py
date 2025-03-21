from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.distributed._shard.checkpoint as dist_cp
from utils import create_and_prepare_model
from transformers import HfArgumentParser, TrainingArguments, set_seed
from peft import get_peft_model, PeftModel
from dataclasses import dataclass, field
from typing import Optional


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
    use_peft_lora: Optional[bool] = field(
        default=True
        )
    use_8bit_quantization: Optional[bool] = field(
        default=False
        )
    use_4bit_quantization: Optional[bool] = field(
        default=False
        )
    use_flash_attn: Optional[bool] = field(
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

parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
model_args, data_args, training_args = parser.parse_args_into_dataclasses()

model, peft_config, tokenizer = create_and_prepare_model(model_args, data_args, training_args, device='cpu')
model = PeftModel.from_pretrained(model, "/home/bbadger/experiments/llama-3.1-8b-codeforcescots-qlora-b64").to('cpu')
print (f"Model pre-merge: {model}")
model = model.merge_and_unload()
print (f"Model post-merge: {model}")
tokenizer.save_pretrained("/home/bbadger/experiments/llama-3.1-8b-codeforcescots-qlora-b64/merged_model")
model.save_pretrained("/home/bbadger/experiments/llama-3.1-8b-codeforcescots-qlora-b64/merged_model")
