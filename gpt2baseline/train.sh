#!/bin/bash
start_time=$(date +%s)

python train_GPT.py --train_set ../dataset/GLUCOSE_train.csv --dev_set ../dataset/GLUCOSE_dev.csv \
    -lr 5e-5 -e 20 --model gpt_baseline.pth --batch_size 16 > gpt_train_log.txt

end_time=$(date +%s)
elpased=$((${end_time}-${start_time}))
echo "GPT-2 training time = $(($elpased/60)) min $(($elpased%60)) sec" > gpt2_train_time.txt
