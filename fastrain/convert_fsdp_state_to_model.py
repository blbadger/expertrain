from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.distributed._shard.checkpoint as dist_cp
import pathlib

modelpath = "/home/badger/llama-3.1-8b-instruct"
model = AutoModelForCausalLM.from_pretrained(
    modelpath,
    torch_dtype=torch.float32,
    device_map="cpu",
)
tokenizer = AutoTokenizer.from_pretrained(modelpath)

state_dict = {
        "model": model.state_dict()
    }

checkpoint_path = pathlib.Path("/home/badger/llama-3.1-8b-cfcots/checkpoint-2400")
distcp_checkpoint_path = checkpoint_path / "pytorch_model_fsdp_0"
dist_cp.load_state_dict(
                state_dict=state_dict,
                storage_reader = dist_cp.FileSystemReader(distcp_checkpoint_path),
                no_dist=True,
            )

model.load_state_dict(state_dict["model"])
out_path = checkpoint_path.parent / "model-2400"
model.save_pretrained(out_path)
tokenizer.save_pretrained(out_path)
