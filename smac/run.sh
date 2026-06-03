#!/bin/bash
alg="ccks"
for n in "MMM" "2s3z"  #n是我们自定义的变量，in后面三个数就是循环3次，每次的值从第一个数的值开始
do

    nohup python3 src/main.py --config=$alg --env-config=sc2 with env_args.map_name=$n env_args.seed=$RANDOM > /dev/null   2>&1   &
    nohup python3 src/main.py --config=$alg --env-config=sc2 with env_args.map_name=$n env_args.seed=$RANDOM > /dev/null   2>&1   &
    nohup python3 src/main.py --config=$alg --env-config=sc2 with env_args.map_name=$n env_args.seed=$RANDOM > /dev/null   2>&1   &
    nohup python3 src/main.py --config=$alg --env-config=sc2 with env_args.map_name=$n env_args.seed=$RANDOM > /dev/null   2>&1   &
    nohup python3 src/main.py --config=$alg --env-config=sc2 with env_args.map_name=$n env_args.seed=$RANDOM > /dev/null   2>&1   &
done