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
parser.add_argument('--gpu_i', type=int)

LLAMA_PROMPT_FORMAT = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>

{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{}<|eot_id|>"""

class QASections:

	def __init__(self, model, chunks, output_path):
		self.model = model
		self.text = text
		self.output_file = output_path
		self.chunks = chunks
		self.qa_outputs = []
		self.unformatted_indices = []


	@classmethod
	def chunk_text_newlines(self, text, batch_size=7):
		chunks = []
		paragraphs = [p for p in text.split('\n') if len(i) > 1]
		for i in range(0, len_paragraphs, page_length):
			chunk = '\n'.join(paragraphs[i:i+page_length])
			chunks.append(chunk)
		return chunks

	@classmethod
	def chunk_text_nearest(self, text, n_char=2000):
		chunks = []
		start = 0
		while start < len(text):
			end = start + n_char
			# continue until newline is found
			while end < len(text) and text[end] != '\n':
				end += 1
			chunks.append(text[start:end])
			start = end + 1 # ignore newline for next text extract
		return chunks

	def generate_qas(self):
		# assumes dataset is loaded in memory
		outputs = []
		for chunk in tqdm(self.chunks):
			print (chunks)
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
				print (chunk, outputs)

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
				arr = []

			for qa_pair in arr:
				question = qa_pair["Question"]
				answer = qa_pair["Answer"]
				formed_string = LLAMA_PROMPT_FORMAT.format(question, answer)
				formatted_outputs.append({'text': formed_string})

		with open(self.output_file, 'w') as f:
			json.dump(formatted_outputs, f)
		print ('json dumped')
		return


if __name__ == '__main__':
	args = parser.parse_args()
	text = open('text_sample.txt', 'r').read()
	chunks = QASections.chunk_text_nearest(text)
	print (len(chunks))
	n_gpus = int(args.n_gpus)
	if n_gpus > 1:
		# divide chunks among GPUs
		gpu_index = int(args.gpu_i)
		selected = int(len(chunks) // n_gpus) 
		remainder = len(chunks) % n_gpus 
		start = gpu_index*selected
		end = gpu_index*selected + selected
		# split remainder chunks evenly among GPUs
		if remainder - gpu_index > 0:
			extra_index = -(remainder - gpu_index)
			extra_chunk = [chunks[extra_index]]
			print (f'GPU {gpu_index}: processing chunks [{start}: {end}] and {extra_index}')
		else:
			extra_chunk = []
			print (f'GPU {gpu_index}: processing chunks [{start}: {end}]')
		selected_chunks = chunks[start: end] + extra_chunk

	print ('Loading model from ', args.model_path)
	model = Llama(
		model_path = args.model_path,
		n_gpu_layers = -1,
		chat_format='llama-3',
		verbose=False,
		n_ctx=8196,
		temperature=0.2
	)

	output_path = '/home/bbadger/experiments/test_qas.json'
	generator = QASections(model, selected_chunks, output_path)
	generator.generate_qas()
	generator.format_qas()

