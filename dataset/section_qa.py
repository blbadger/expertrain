from llama_cpp import Llama
import json
from datasets import load_dataset, Dataset
from tqdm import tqdm
import torch
import argparse
import ast
import re

# Instantiate the parser
parser = argparse.ArgumentParser(description='Optional app description')
parser.add_argument('--n_gpus', type=int)
parser.add_argument('--model_path', type=str)
parser.add_argument('--gpu_i', type=int)
parser.add_argument('--output_path', type=str)

LLAMA_PROMPT_FORMAT = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>

{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{}<|eot_id|>"""

class QASections:

	def __init__(self, model, chunks):
		self.model = model
		self.text = text
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
			if len(chunk) > 1:
				output = model.create_chat_completion(
			      messages = [
					{"role": "system", "content": "You are helpful assistant responsible for creating good questions from text and answering them."},
				        {
			            "role": "user",
			            "content": f"""
							Given the following Context, give five insightful questions about the text and answer each one accurately in the following JSON format: 
							[
								{{"Question": "[insert question]", "Answer": "[insert answer]"}},
								{{"Question": "[insert question]", "Answer": "[insert answer]"}},
								...
							]		

							Answer in valid JSON with no other text. Do not begin your answer with a phrase like 'Here are five insightful questions...'.

							Context:
							{chunk}
							"""
					    }
					]
				)
				outputs.append(output["choices"][0]["message"]["content"])

		self.qa_outputs = outputs
		return outputs

	def format_qas(self):
		formatted_outputs = []
		for i, string in enumerate(self.qa_outputs):
			question_chars = len('"Question": ')
			answer_chars = len('"Answer": ')
			all_questions = [k.start() for k in re.finditer('"Question": ', string)]
			all_answers = [k.start() for k in re.finditer('"Answer": ', string)]

			for i, pair in enumerate(zip(all_questions, all_answers)):
				# if on the last qa pair
				if i == len(all_questions) - 1:
					answer_stop = len(string)
				else:
					answer_stop = all_questions[i+1]
				question = string[pair[0]+question_chars:pair[1]]
				answer = string[pair[1]+answer_chars:answer_stop]
				question = question.strip(',"}\n{" ][')
				answer = answer.strip('",}\n{" ][')
				formed_string = LLAMA_PROMPT_FORMAT.format(question, answer)
				formatted_outputs.append({"text": formed_string})

		return formatted_outputs


if __name__ == '__main__':
	args = parser.parse_args()
	text = open('data/github_pages/all_pages.md', 'r').read()
	print ('Loading model from ', args.model_path)
	model = Llama(
			model_path = args.model_path,
			n_gpu_layers = -1,
			chat_format='llama-3',
			verbose=False,
			n_ctx=8196,
			temperature=0.2 # generally should be low for factual q/a in semi-correct JSON
		)
	# if more than one char limit given, none should be multiples of any other
	char_limits = [1000, 2500, 6500, 10500]
	all_outputs = []
	for char_lim in char_limits:
		chunks = QASections.chunk_text_nearest(text, n_char=char_lim)
		print ('Chunks to process: ', len(chunks))
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
				print (f'GPU {gpu_index}: processing chunks of indices [{start}: {end}) and {len(chunks)+extra_index}')
			else:
				extra_chunk = []
				print (f'GPU {gpu_index}: processing chunks of indices [{start}: {end})')
			selected_chunks = chunks[start: end] + extra_chunk

		generator = QASections(model, selected_chunks)
		generator.generate_qas()
		formatted_outputs = generator.format_qas()
		all_outputs += formatted_outputs

	output_path = args.output_path + f'_{gpu_index}' # add identifier for each thread
	dataset = Dataset.from_list(all_outputs)
	dataset.save_to_disk(output_path)
	print ('Outputs saved')

