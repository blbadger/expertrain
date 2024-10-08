import os
import re
import shutil

root_dir = '/home/bbadger/Desktop/tinystories/test_clean'

for fp in os.listdir(root_dir):
	full_filepath = root_dir + '/' + fp
	if not os.path.isdir(full_filepath) or str(fp) == 'landscapes':
		continue
	dirs = os.listdir(full_filepath)
	dir_pairs = []
	for d in dirs:
		if 'checkpoint' not in str(d): 
			continue 
		match = re.findall('\-.*', d)
		if len(match) == 0: 
			continue
		number = int(str(match[0]).strip('_'))
		dir_pairs.append([d, number])

	dir_pairs.sort(key = lambda x: x[1])
	sorted_dirs = [i[0] for i in dir_pairs] # checkpoints last to first
	print (sorted_dirs)
	# keep last checkpoint
	for dir in sorted_dirs[1:]:
		print (full_filepath + '/' + dir, ' directory removed')
		shutil.rmtree(full_filepath + '/' + dir)

