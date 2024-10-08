from llama_cpp import Llama
import json
from datasets import load_dataset
from tqdm import tqdm
import torch
import argparse

# Instantiate the parser
parser = argparse.ArgumentParser(description='Optional app description')
parser.add_argument('--n_gpus', type=int)
parser.add_argument('--model_path', type=str)
parser.add_argumnet('--gpu_i', type=str)
print ('parser initialized')

PROMPT_FORMAT = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>

{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{}<|eot_id|>"""

class QASections:

	def __init__(self, model, text, output_path):
		self.model = model
		self.text = text
		self.output_file = output_path
		self.chunks = []
		self.qa_outputs = []
		self.unformatted_indices = []

	def chunk_text(self):
		for paragraph in self.text.split('\n'):
			if len(paragraph) > 1:
				self.chunks.append(paragraph)
		return

	def generate_qas(self):
		# assumes dataset is loaded in memory
		outputs = []
		for chunk in tqdm(self.chunks):
			if len(chunk) > 1:
				output = model.create_chat_completion(
			      messages = [
					{"role": "system", "content": "You are helpful assistant responsible for creating good questions from text and answering them."},
				        {
			            "role": "user",
			            "content": f"""
							Given the following Context, give five insightful questions about the text and answer each one accurately in the following JSON format: 
							
							{{"Question": "[insert question]", "Answer": "[insert answer]"}}

							Answer in valid JSON with no other text.

							Context:
							{chunk}
							"""
					    }
					]
				)
				# print (chunk, output)
				outputs.append(output["choices"][0]["message"]["content"])
		self.qa_outputs = outputs
		return outputs

	def format_qas(self):
		formatted_outputs = []
		for i, json_string in enumerate(self.qa_outputs):
			# add final brace if necessary
			if json_string[-1] != '}':
				json_string += '}'

			try:
				arr = list(json.loads('[' + json_string + ']'))
			except:
				self.unformatted_indices.append(i)

			for qa_pair in arr:
				question = qa_pair["Question"]
				answer = qa_pair["Answer"]
				formed_string = PROMPT_FORMAT.format(question, answer)
				formatted_outputs.append({'text': formed_string})

		with open(self.output_file, 'w') as f:
			json.dump(formatted_outputs, f)
		return


if __name__ == '__main__':
	args = parser.parse_args()
	n_gpus = args.n_gpus
	if n_gpus > 1:
		# divide chunks among GPUs
		gpu_index = torch.cuda.current_device()
		selected = len(self.chunks) // n_gpus
		selected_chunks = self.chunks[gpu_index*selected: gpu_index*selected+selected]

	print ('Loading model from ', args.model_path)
	model = Llama(
		model_path = args.model_path,args = parser.parse_args(
		n_gpu_layers = -1,
		chat_format='llama-3',
		verbose=False,
		n_ctx=8196,
		temperature=0.2
	)

	output_path = '/home/bbadger/experiments/test_qas.json'
	generator = QASections(model, text, output_path)
	generator.chunk_text()
	generator.generate_qas()
	generator.format_qas()





