import datasets

def string_to_dataset(text_path, output_path):
	text = str(open(path))
	data = [{"text": text}]
	dataset = dataset.Dataset.from_list(data)
	dataset.save_to_disk()

if __name__ == "__main__":
	path = 'data/github/all_pages.md'
	output_path = 'data/github/all_pages_dataset'
	string_to_dataset(text_path, output_path)
