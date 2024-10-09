import argparse
import subprocess
import torch

template = "CUDA_VISIBLE_DEVICES={} section_qa.py --n_gpus {} --model_path {} &\n"

# Instantiate the parser
parser = argparse.ArgumentParser(description='Optional app description')
parser.add_argument('--n_gpus', type=int)
parser.add_argument('--model_path', type=str)
print ('parser initialized')

if __name__ == "__main__":
	args = parser.parse_args()
	n_gpus = torch.cuda.device_count()
	bash_string = ""
	for gpu_index in range(n_gpus):
		bash_string += template.format(gpu_index, n_gpus, model_path)

	print (f'Running string: {bash_string}')
	subprocess.run(bash_string, shell=True)

