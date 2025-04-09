from unsloth import FastLanguageModel
import torch
import re
from trl import GRPOConfig, GRPOTrainer
from datasets import load_dataset, load_from_disk, Dataset
import sqlite3

# Load and prep dataset
SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""

max_seq_length = 2048 # Can increase for longer reasoning traces
lora_rank = 32 # Larger rank = smarter, but slower

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "meta-llama/meta-Llama-3.1-8B-Instruct",
    max_seq_length = max_seq_length,
    load_in_4bit = True, # False for LoRA 16bit
    fast_inference = True, # Enable vLLM fast inference
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.6, # Reduce if out of memory
)

model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ], # Remove QKVO if out of memory
    lora_alpha = lora_rank,
    use_gradient_checkpointing = "unsloth", # Enable long context finetuning
    random_state = 3407,
)

def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

# uncomment middle messages for 1-shot prompting
def get_gsm8k_questions(split = "train") -> Dataset:
    data = load_dataset('openai/gsm8k', 'main')[split] # type: ignore
    data = data.map(lambda x: { # type: ignore
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['question']}
        ],
        'answer': extract_hash_answer(x['answer']),
        'database': 'yes'
    }) # type: ignore
    return data # type: ignore

dataset = get_gsm8k_questions()
print (dataset[0])

def get_bird_dataset():
    
    train_dataset = load_from_disk('/home/bbadger/Desktop/birds/bird/llm/data/train_dataset_non-prefilled')
    train_dataset = train_dataset.map(lambda x: {
         'prompt': [
             {'role': 'user', 'content': x['messages'][0]['content']},
          ],
          'answer': x['messages'][1]['content'],
          'database': x['databases']
    }, remove_columns=['messages'])
    eval_dataset = load_from_disk('/home/bbadger/Desktop/birds/bird/llm/data/dev_dataset_non-prefilled')
    eval_dataset = eval_dataset.map(lambda x: {
         'prompt': [
             {'role': 'user', 'content': x['messages'][0]['content']},
          ],  
          'answer': x['messages'][1]['content'],
          'database': x['databases']
    }, remove_columns=['messages'])  
    train_dataset = train_dataset.filter(lambda x: len(tokenizer.encode(x['prompt'][0]['content'])) < max_seq_length)
    return train_dataset, eval_dataset

train_dataset, eval_dataset = get_bird_dataset()
print (train_dataset[0])
print (len(train_dataset))
model_path=''

def bird_check(predicted_sql, ground_truth, db_path):
    """
    Check the output for execution accuracy.
    Args:
        output (str): The generated SQL query.
        gold_sql (str): The ground truth SQL query.
    Returns:
        bool: True if the output is correct, False otherwise.
    """
    full_db_path = '/home/bbadger/Desktop/birds/bird/llm/data/train_databases' + '/' + f'{db_path}/{db_path}.sqlite'
    #print (full_db_path, predicted_sql, ground_truth)
    conn = sqlite3.connect(full_db_path)
    # Connect to the database
    cursor = conn.cursor()
    try:
        cursor.execute(predicted_sql)
        predicted_res = cursor.fetchall()
    except Exception:
        return 0 # malformed sql
    cursor.execute(ground_truth)
    ground_truth_res = cursor.fetchall()
    res = 0
    if set(predicted_res) == set(ground_truth_res):
        res = 1
    return res

# Reward functions
def correctness_reward_func(prompts, completions, answer, database, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

def execution_reward_func(prompts, completions, answer, database, cot=True, **kwargs):
    responses = [completion[0]['content'] for completion in completions]
    outputs = []
    for response in responses:
        if cot:
            output = extract_xml_answer(response)
        else:
            output = response
        outputs.append(output)

    gold_sqls = [sql for sql in answer]
    databases = [db for db in database]

    checks = []
    for output, gold, db in zip(outputs, gold_sqls, databases):
        check = bird_check(output, gold, db)
        checks.append(check)
    return [1.0 if completion_check else 0.0 for completion_check in checks]

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def length_reward(completions, **kwargs):
    """Reward function for increased response length."""
    responses = [completion[0]["content"] for completion in completions]
    return [0.001*len(response) for response in responses]

def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1])*0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]

max_prompt_length = 1900

training_args = GRPOConfig(
    learning_rate = 5e-6,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type = "cosine",
    optim = "paged_adamw_8bit",
    logging_steps = 1,
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 2, # Increase to 4 for smoother training
    num_generations = 4, # Decrease if out of memory
    max_prompt_length = max_prompt_length,
    max_completion_length = max_seq_length - max_prompt_length,
    num_train_epochs = 1, # Set to 1 for a full training run
    save_steps = 500,
    max_grad_norm = 0.1,
    report_to = "none", # Can use Weights & Biases
    output_dir = "outputs",
)

trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [
        #xmlcount_reward_func,
        #soft_format_reward_func,
        #strict_format_reward_func,
        #int_reward_func,
        execution_reward_func,
    ],
    args = training_args,
    train_dataset = train_dataset,
   #eval_dataset = eval_dataset
)
trainer.train()
