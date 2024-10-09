CUDA_VISIBLE_DEVICES=0 python section_qa.py --n_gpus 4 --gpu_i 0 --model_path /home/bbadger/Desktop/llama-3-8b-instruct-Q8_0.gguf &
CUDA_VISIBLE_DEVICES=1 python section_qa.py --n_gpus 4 --gpu_i 1 --model_path /home/bbadger/Desktop/llama-3-8b-instruct-Q8_0.gguf &
CUDA_VISIBLE_DEVICES=2 python section_qa.py --n_gpus 4 --gpu_i 2 --model_path /home/bbadger/Desktop/llama-3-8b-instruct-Q8_0.gguf &
CUDA_VISIBLE_DEVICES=3 python section_qa.py --n_gpus 4 --gpu_i 3 --model_path /home/bbadger/Desktop/llama-3-8b-instruct-Q8_0.gguf