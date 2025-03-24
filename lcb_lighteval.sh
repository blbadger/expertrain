MODEL=/home/bbadger/experiments/llama-3.1-8b-codeforcescots
MODEL_ARGS="pretrained=$MODEL,tokenizer=$MODEL,max_length=32768,dtype=float16"
OUTPUT_DIR=data/evals/$MODEL

lighteval accelerate $MODEL_ARGS "extended|lcb:codegeneration|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR \
    --override-batch-size 1 
