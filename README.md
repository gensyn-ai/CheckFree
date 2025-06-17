Official repo of the All is not Lost Paper.

# All is Not Lost (AKA CheckFfree)

A novel way of recovering weights after a fault, without relying on checkpointing or additional computation, which improves the training time compared to the state of the art by over 10%.

- **Novel checkpoint-less recovery:** CheckFree uses the weights of neighbouring stages to approximate the weights of the lost stage.


- **200% speed up to conventional checkpointing:** CheckFree and CheckFree+ can achieve up to 200% training time speed up compared to conventional checkpointing in the presence of frequent stage faults.

# Setup

The repository depends on two libraries:

- [simplellm](https://github.com/NikolayBlagoev/simplellm) - for construction of the models, loading datasets, tokenizers, etc.

- [DecCom](https://github.com/NikolayBlagoev/DecCom-Python) - for communication between devices

You can install both by cloning the repo and doing ```pip install .``` or by running the [setup.sh](/setup.sh) provided here.

Additionally, you need to install the requirements in [requirements.txt](/requirements.txt) with ```pip install -r requirements.txt```

# Simulated training

To run the simulated fault training run one of the files in [simulate_training](/simulate_training/) like:

```
./run.sh no_failure 4 10 500M_config.json 0
```

The scripts have form:
```
./run.sh [SETTING] [WORLD_SIZE] [FAILURE PROBABILITY] [MODEL CONFIG] 0
```

You can use [run.sh](/simulate_training/run.sh) for training with swaps and [run_2.sh](/simulate_training/run_2.sh) for training without swaps

All training scripts:

```
./run.sh no_failure 4 10 500M_config_gpt.json 0
./run.sh no_failure 2 10 124M_config.json 0
./run.sh ours-grad-avg 4 16 500M_config_gpt.json 0


./run.sh no_failure 8 16 1_5B_config.json 100
./run.sh ours-naive 4 10 500M_config.json 0
./run.sh ours-random 4 10 500M_config.json 0
./run.sh ours-grad-avg 2 10 124M_config.json 0
./run.sh ours-grad-avg 8 16 1_5B_config.json 0
./run.sh ours-grad-avg 4 33 500M_config.json 0

./run.sh ours-grad-avg 4 10 500M_config.json 0
./run.sh ours-grad-avg 4 16 500M_config.json 0
./run.sh ours-grad-avg 4 5 500M_config.json 0
./run.sh ours-grad-avg-regularize 4 16 500M_config.json 0
./run.sh ours-grad-avg 4 33 500M_config.json 0

./run.sh ours-zero 4 16 500M_config.json 0


./run_2.sh ours-grad-avg 4 5 500M_config.json 0
./run_2.sh ours-grad-avg 4 10 500M_config.json 0
./run_2.sh ours-grad-avg 4 16 500M_config.json 0
./run_2.sh ours-grad-avg 2 10 124M_config.json 0
```



# Throughput tests

To evaluate throughput of different strategies, use the scripts in [communication/](/communication/)

You can run a given test via [run_throughput.sh](/run_throughput.sh):

```
./run_throughput.sh [STARTING NODE] [END NODE] [SETTING] [FAILURE RATE]
```

Failure rate defines the config file in [failure_p_configs](/failure_p_configs/) to use (showing which node crashes at which iteration)

You can generate new configs via [failure_generator.py](/failure_generator.py)

