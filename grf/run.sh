#!/bin/bash
alg="cola_cons"
for n in "academy_run_pass_and_shoot_with_keeper" 
# "academy_3_vs_1_with_keeper" "academy_corner" "academy_pass_and_shoot_with_keeper" "academy_counterattack_hard" "academy_run_pass_and_shoot_with_keeper"
do
    # nohup python3 src/main.py --config=$alg --env-config=sc2 with env_args.map_name=$n env_args.seed=520 > /dev/null   2>&1   &
    # nohup python3 src/main.py --config=$alg --env-config=sc2 with env_args.map_name=$n env_args.seed=1 > /dev/null   2>&1   &
    # nohup python3 src/main.py --config=$alg --env-config=sc2 with env_args.map_name=$n env_args.seed=10 > /dev/null   2>&1   &
    # nohup python3 src/main.py --config=$alg --env-config=sc2 with env_args.map_name=$n env_args.seed=100 > /dev/null   2>&1   &
    nohup python3 src/main.py --config=$alg --env-config=sc2 with env_args.map_name=$n env_args.seed=1000 > /dev/null   2>&1   & 
    nohup python3 src/main.py --config=$alg  --env-config=$n with seed=$RANDOM > /dev/null   2>&1   &
    nohup python3 src/main.py --config=$alg  --env-config=$n with seed=$RANDOM > /dev/null   2>&1   &
    nohup python3 src/main.py --config=$alg  --env-config=$n with seed=$RANDOM > /dev/null   2>&1   &
    nohup python3 src/main.py --config=$alg  --env-config=$n with seed=$RANDOM > /dev/null   2>&1   &
    nohup python3 src/main.py --config=$alg  --env-config=$n with seed=$RANDOM > /dev/null   2>&1   &
done