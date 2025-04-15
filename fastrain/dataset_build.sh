source ../.env
model_path=''
eval_path='./data/train.json'
dev_path='./output/'
db_root_path='./data/train_databases/'
use_knowledge='True'
mode='train' # choose dev or train
cot='False'

echo 'Building Dataset'
python3 -u ./src/dataset_builder.py \
    --model_path ${model_path} \
    --db_root_path ${db_root_path} \
    --mode ${mode} \
    --eval_path ${eval_path} \
    --use_knowledge ${use_knowledge} \
    --chain_of_thought ${cot}

