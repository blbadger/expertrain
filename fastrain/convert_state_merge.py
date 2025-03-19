from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.distributed._shard.checkpoint as dist_cp
from utils import create_and_prepare_mdoel
from transformers import HfArgumentParser, TrainingArguments, set_seed
from peft import get_peft_model

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

parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
model_args, data_args, training_args = parser.parse_args_into_dataclasses()

model, peft_config, tokenizer = create_and_prepare_model(model_args, data_args, training_args, device='cpu')
peft_model = get_peft_model(model, peft_config)

state_dict = {
        "model": model.state_dict()
    }

distcp_checkpoint_path = "/home/bbadger/experiments/llama-3.1-8b-codeforcescots-qlora/checkpoint-5060/pytorch_model_fsdp_0"
dist_cp.load_state_dict(
                state_dict=state_dict,
                storage_reader = dist_cp.FileSystemReader(distcp_checkpoint_path),
                no_dist=True,
            )

model.load_state_dict(state_dict["model"])
model.save_pretrained("/home/bbadger/experiments/llama-3.1-8b-codeforcescots-qlora/model")
