
## Table of Contents

- [Yuan3.0 Flash](#Yuan30-Flash)
  - [Table of Contents](#table-of-contents)
  - [Quick Start](#quick-start)
    - [Environment Config](#environment-config)
    - [Data preprocess](#data-preprocess)
    - [Model Instruct-tuning](#instruct-tuning)
    - [Reinforce Learning](#reinforce-learning)
  - [Inference Service](#inference-service)
    - [linux Deployment](#linux-service)
  - [contact use](#contact_us)

# Yuan3.0 Flash

-----

The use of the source code in this repository requires compliance with the open source license agreement **Apache 2.0**.
The Yuan3.0 Flash model supports commercial use and does not require authorization. Please understand and comply with the [《Yuan3.0 Flash Model License Agreement》](./LICENSE-Yuan). Do not use the open source model and code, as well as derivatives generated from open source projects, for any purposes that may cause harm to the country and society, or for any services that have not undergone security assessment and filing.
Although we have taken measures to ensure the compliance and accuracy of the data during training, the model has a huge number of parameters and is affected by probability and randomness factors. We cannot guarantee the accuracy of the output content, and the model is easily misled by input instructions. This project does not assume any data security, public opinion risks, or any model misleading, abusing, spreading caused by open-source models and code Risks and responsibilities arising from improper utilization  **You will be solely responsible for the risks and consequences arising from the use, copying, distribution, and modification of the model in this open source project**

## Quick Start

### Environment Config

We recommend using the latest pre-built docker image provided by us.

We can start the container with the following command:
```bash
docker pull yuanlabai/rlhf_yuan:v1.0

docker run --gpus all -itd --network=host -v /path/to/yuan_3.0:/workspace/yuan_3.0 -v /path/to/dataset:/workspace/dataset -v /path/to/checkpoints:/workspace/checkpoints --cap-add=IPC_LOCK --device=/dev/infiniband --privileged --name your_name --ulimit core=0 --ulimit memlock=-1 --ulimit stack=68719476736 --shm-size=1000G yuanlabai/rlhf_yuan:v1.0

docker exec -it your_name bash
```

We only support using the provided docker image.

#### SQL Environment Configuration

##### Dataset
1. **Download Links**:
   - Spider: https://drive.google.com/file/d/1mkCx2GOFIqNesD4y8TDAO1yX1QZORP5w/view
   - BIRD: https://bird-bench.github.io/

2. **Configuration Commands** (execute on head node):
```bash
# Set local dataset paths
export BIRD_PATH=/your_local_path/bird_data
export SPIDER_PATH=/your_local_path/spider_data

# Copy to working directory
cp -rf $BIRD_PATH /home/data/
cp -rf $SPIDER_PATH /home/
```


### Data preprocess

We have provided the data preprocess script. See documentation [here](./docs/data_process.md).

### Model Instruct Tuning

We also have provided the supervised fine-tuning script. See documentation [here](./docs/instruct_tuning.md).


### Reinforcment Learning


We provide the process of reinforcement learning, see documentation [here](./docs/RL_training.md).


### Inference Deployment

#### linux Deployment

vllm deployment process, see documentation [here](../vllm/README_Yuan.md)

### Contact Us

Contact us: service@yuanlab.ai
