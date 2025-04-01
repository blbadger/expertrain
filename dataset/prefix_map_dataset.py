from datasets import load_from_disk
import torch
from multiprocess import set_start_method
from transformers import AutoTokenizer, AutoModelForCausalLM 
from datasets import load_dataset

dataset = load_from_disk('/home/bbadger/experiments/bird_train_dataset')

def add_prefix(sample):
	sample["messages"][1]["content"] = '###' + sample["messages"][1]["content"]
	return sample


updated_dataset = dataset.map(add_prefix)
print (updated_dataset[0])
updated_dataset.save_to_disk("/home/bbadger/experiments/bird_train_dataset_completion")
