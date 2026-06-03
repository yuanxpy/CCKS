# CCKS——**Consensus-based Communication and Knowledge Sharing**

## Introduction
CCKS is the first framework to enhance decentralized training and execution (DTDE) in multi-agent reinforcement learning (MARL) by introducing a consensus-based communication and knowledge-sharing mechanism. We believe the CCKS framework is highly effective, significantly improving cooperation efficiency and overall performance with minimal modifications to existing algorithms. Our approach of integrating consensus learning and knowledge sharing through communication can be seamlessly applied to any DTDE algorithm.

Note: Our code is modified based on [COLA](https://github.com/deligentfool/COLA) and [PyMARL](https://github.com/oxwhirl/pymarl). 

## Run an experiment 

AN EXAMPLE For SMAC

```shell
python3 src/main.py --config=ccks --env-config=sc2 with env_args.map_name=2s3z env_args.seed=1
```

AN EXAMPLE FOR GRF
```shell
python3 src/main.py --config=ccks --env-config=academy_3_vs_1_with_keeper with seed=1
```

`--config` refers to the algorithm config `
--env-config` refers to the scenario

The config files is located in `src/config` floder and the result after running will be stored in the `Results` folder.

Or you can use our provided `run.sh`, which only need to replace `alg` and `n`  .
