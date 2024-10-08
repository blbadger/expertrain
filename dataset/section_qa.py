from llama_cpp import Llama
import json
from datasets import load_dataset
from tqdm import tqdm
import torch
import argparse

# Instantiate the parser
parser = argparse.ArgumentParser(description='Optional app description')
#parser.add_argument('--n_gpus', type=int)
parser.add_argument('--model_path', type=str)
print ('parser initialized')
# n_gpus = args.n_gpus
# if n_gpus > 1:
# 	# divide chunks among GPUs
# 	n_gpus = torch.cuda.device_count()
# 	gpu_index = torch.cuda.current_device()
# 	selected = len(self.chunks) // n_gpus
# 	selected_chunks = self.chunks[gpu_index*selected: gpu_index*selected+selected]

PROMPT_FORMAT = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>

{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{}<|eot_id|>"""

class QASections:

	def __init__(self, model, text):
		self.model = model
		self.text = text
		self.chunks = []
		self.qa_outputs = []

	def chunk_text(self):
		for paragraph in self.text.split('\n'):
			if len(paragraph) > 1:
				self.chunks.append(paragraph)
		return

	def generate_qas(self):
		# assumes dataset may be loaded in memory
		#text = load_dataset(path)
		outputs = []
		for chunk in tqdm(self.chunks):
			if len(chunk) > 1:
				output = model.create_chat_completion(
				      messages = [
						{"role": "system", "content": "You are helpful assistant responsible for creating good questions from text and answering them."},
					        {
					            "role": "user",
					            "content": f"""
									Given the following Context, give five insightful questions about the text and answer each one accurately in the following JSON format: {{"Question", "Answer"}}. Answer in valid JSON with no other text.

									Context:
									{chunk}
								"""
						        }
							]
						)
				print (chunk, output)
				outputs.append(output)
		self.qa_outputs = outputs
		return outputs

	def format_qas(self):
		qa_pairs = []
		for arr in self.qa_outputs:
			qa_pairs += arr
		print (qa_pairs)

		formatted_outputs = []
		for question, answer in qa_pairs:
			formed_string = PROMPT_FORMAT.format(question, answer)
			formatted_outputs.append({'text': formed_string})
		with open(output_file, 'w') as f:
			json.dump(formatted_outputs, f)
		return


if __name__ == '__main__':
	args = parser.parse_args()
	print ('Loading model from ', args.model_path)
	model = Llama(
		model_path = args.model_path,
		n_gpu_layers = -1,
		chat_format='llama-3',
		verbose=False,
		n_ctx=8196,
		temperature=0.3
	)

	text = """ Washington, D.C., formally the District of Columbia and commonly known as Washington or D.C., is the capital city and federal district of the United States. The city is on the Potomac River, across from Virginia, and shares land borders with Maryland to its north and east. It was named for George Washington, the first president of the United States. The district is named for Columbia, the female personification of the nation.

The U.S. Constitution in 1789 called for the creation of a federal district under the exclusive jurisdiction of the U.S. Congress. As such, Washington, D.C., is not part of any state, and is not one itself. The Residence Act, adopted on July 16, 1790, approved the creation of the capital district along the Potomac River. The city was founded in 1791, and the 6th Congress held the first session in the unfinished Capitol Building in 1800 after the capital moved from Philadelphia. In 1801, the District of Columbia, formerly part of Maryland and Virginia and including the existing settlements of Georgetown and Alexandria, was officially recognized as the federal district; initially, the city was a separate settlement within the larger district. In 1846, Congress returned the land originally ceded by Virginia, including the city of Alexandria. In 1871, it created a single municipality for the remaining portion of the district. There have been several unsuccessful efforts to make the district into a state since the 1880s; a statehood bill passed the House of Representatives in 2021 but was not adopted by the U.S. Senate.

Designed in 1791 by Pierre Charles L'Enfant, the city is divided into quadrants, which are centered around the Capitol Building and include 131 neighborhoods. As of the 2020 census, the city had a population of 689,545.[3] Commuters from the city's Maryland and Virginia suburbs raise the city's daytime population to more than one million during the workweek.[12] The Washington metropolitan area, which includes parts of Maryland, Virginia, and West Virginia, is the country's seventh-largest metropolitan area, with a 2023 population of 6.3 million residents.[6] A locally elected mayor and 13-member council have governed the district since 1973, though Congress retains the power to overturn local laws. Washington, D.C., residents are, on the federal level, politically disenfranchised since the city's residents do not have voting representation in Congress; the city's residents elect a single non-voting congressional delegate to the U.S. House of Representatives. The city's voters choose three presidential electors in accordance with the Twenty-third Amendment. """

	generator = QASections(model, text)
	generator.chunk_text()
	generator.generate_qas()
	generator.format_qas()





