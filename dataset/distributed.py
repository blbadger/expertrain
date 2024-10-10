import argparse
import subprocess
import torch

template = "CUDA_VISIBLE_DEVICES={} python section_qa.py --n_gpus {} --gpu_i {} --model_path {} --output_path {}&\n"

# Instantiate the parser
parser = argparse.ArgumentParser(description='Optional app description')
parser.add_argument('--n_gpus', type=int)
parser.add_argument('--model_path', type=str)
parser.add_argument('--output_path', type=str)
print ('parser initialized')

if __name__ == "__main__":
	args = parser.parse_args()
	n_gpus = args.n_gpus
	model_path = args.model_path
	output_path = args.output_path
	bash_string = ""
	for gpu_index in range(n_gpus):
		bash_string += template.format(gpu_index, n_gpus, gpu_index, model_path, output_path)
	bash_string = bash_string[:-3] # strip _&/n from last templated entry
	print (f'Running string: {bash_string}')
	subprocess.run(bash_string, shell=True)

