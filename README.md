# Minimum Cost Human-Machine Active Labeling

Prerequisite: Tensorflow-2.4.1, CUDA-11, Keras 2.3.1

## Running in Docker

Follow instructions to install latest [docker-ce](https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository), and [nvidia-docker2](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)


```
# Building docker image locally
cd Docker/TF-2.4_Keras-2.3.1
sudo docker build -t hangqiu/mcal:tf2.4 .
# or Pulling docker
sudo docker pull hangqiu/mcal:tf2.4

# Running in docker

sudo docker run -u $(id -u):$(id -g)  -it --rm --gpus=all --ipc=host -v ~/research/ActiveLabeling:/ActiveLabeling  -v ~/research/dataset:/dataset hangqiu/mcal:tf2.4

```

## Active Learning Baseline

```bash
# An example parameter set:
TRIAL=1
HORIZON=0.99
GPU=0,1,2,3
CELL=3
LAYER=3
KERNEL=16
BATCHSIZE=1000
WARMSTARTSIZE=SAME
AUG=1
ACCThresh=0.95
MINIBATCH=256
DATASET=cifar10_keras
MODEL=resnet_grow

python3 run_experiment.py \
    --dataset=${DATASET} \
    --score_method=${MODEL} \
    --standardize_data=False \
    --train_horizon=${HORIZON} \
    --trials=${TRIAL} \
    --sampling_method=${SAMPLE} \
    --data_dir=../dataset/tf-data \
    --gpu=${GPU} \
    --cell=${CELL} \
    --layer=${LAYER} \
    --kernel=${KERNEL} \
    --warmstart_size=${WARMSTARTSIZE} \
    --batch_size=${BATCHSIZE}\
    --augmentation=${AUG}\
    --accthresh=${ACCThresh}\
    --minibatch=${MINIBATCH}

```

## MCAL (Active Labeling)
```bash
python3 optimized_labeling.py \
    --dataset=${DATASET} \
    --score_method=${MODEL} \
    --standardize_data=False \
    --train_horizon=${HORIZON} \
    --trials=${TRIAL} \
    --sampling_method=${SAMPLE} \
    --data_dir=../dataset/tf-data \
    --gpu=${GPU} \
    --cell=${CELL} \
    --layer=${LAYER} \
    --kernel=${KERNEL} \
    --warmstart_size=${WARMSTARTSIZE} \
    --batch_size=${BATCHSIZE}\
    --augmentation=${AUG}\
    --accthresh=${ACCThresh}\
    --minibatch=${MINIBATCH}
 ```
 
 ## Evaluation
 ```bash
 sh run_active_learning.sh
 sh run_optimized_labeling.sh
 ```

## Citation

```bibtex
@inproceedings{
qiu2023mcal,
title={{MCAL}: Minimum Cost Human-Machine Active Labeling},
author={Hang Qiu and Krishna Chintalapudi and Ramesh Govindan},
booktitle={International Conference on Learning Representations},
year={2023},
url={https://openreview.net/forum?id=1FxRPKrH8bw}
}
```

Note: This repo is developed based on [google/active-learning](https://github.com/google/active-learning).