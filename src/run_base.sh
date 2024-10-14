#!/bin/bash
export CUDA_VISIBLE_DEVICES=1,3
export CUDA_LAUNCH_BLOCKING=1


domains=('lap14')
data='../data/'

for domain in  ${domains[@]};
do
    echo "####################### ${model_name} ${domain} #######################:"
    python -B run_base.py \
        --plm_model bert_base  --seed 42  \
        --data_dir "${data}${domain}"  --batch_size 8  --epochs 15 --learning_rate 2e-5 --max_seq_length 80  \
        --do_train --do_eval  --logging_global_step 30 \
        --demons_pool_size 100  --shot_num 4  --set_num 8
    printf "\n\n"

done