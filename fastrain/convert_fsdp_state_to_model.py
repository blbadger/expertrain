from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.distributed._shard.checkpoint as dist_cp

modelpath = "/home/bbadger/Desktop/llama-3.1-8b-instruct"
model = AutoModelForCausalLM.from_pretrained(
    modelpath,
    torch_dtype=torch.float32,
    device_map="cpu",
)
tokenizer = AutoTokenizer.from_pretrained(modelpath)

state_dict = {
        "model": model.state_dict()
    }

distcp_checkpoint_path = "/home/bbadger/experiments/llama-3.1-8b-codeforcescots-qlora/checkpoint-2000/pytorch_model_fsdp_0"
dist_cp.load_state_dict(
                state_dict=state_dict,
                storage_reader = dist_cp.FileSystemReader(distcp_checkpoint_path),
                no_dist=True,
            )

model.load_state_dict(state_dict["model"])
model.save_pretrained("/home/bbadger/experiments/llama-3.1-8b-codeforcescots-qlora/model")
