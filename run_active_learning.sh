#!/bin/bash


TRIAL=1
HORIZON=0.99
GPU=0,1,2,3
#GPU=0,1

CELL=3
LAYER=3
KERNEL=16

#BATCHSIZE=16000,8000,6000,4000,2000,1000,10000
#BATCHSIZE=4000
#BATCHSIZE=32000,16000
#BATCHSIZE=1000
# BATCHSIZE=500
# BATCHSIZE=6000,10000
# BATCHSIZE=1.0,0.4,0.2,0.1,0.05,0.025,0.0125,0.00625,0.003125
# BATCHSIZE=0.003125,0.0015625,0.00078125
# BATCHSIZE=0.003125
BATCHSIZE=0.5
# BATCHSIZE=0.4,0.2,0.1,0.05,0.025,0.0125,0.00625
# BATCHSIZE=8000,2000,1000,16000,500,250
# BATCHSIZE=2000
# BATCHSIZE=50

WARMSTARTSIZE=SAME
# WARMSTARTSIZE=4000

AUG=1

#ACCThresh=0.9
ACCThresh=0.95

#SAMPLE=margin
SAMPLE=uniform
#SAMPLE=kcenter
#SAMPLE=entropy
#SAMPLE=least_confidence
#SAMPLE=graph_density

#LABELING_SAMPLE=margin
#LABELING_SAMPLE=uniform
LABELING_SAMPLE=kcenter
#LABELING_SAMPLE=entropy
#LABELING_SAMPLE=least_confidence

#MINIBATCH=32
MINIBATCH=64
#MINIBATCH=128
#MINIBATCH=256
#MINIBATCH=512

#DATASET=imagenet
DATASET=image_path_dataset
DATASET_DIR=../dataset/ImageNet/raw_data/train/
# DATASET=tinyimagenet
#DATASET=cifar10_keras
#DATASET=cif12_keras
#DATASET=cif24_keras
#DATASET=cif36_keras
#DATASET=cif48_keras
#DATASET=cifar100_keras
# DATASET=svhn
# DATASET=fashion_keras
# DATASET=mnist_keras
#MODEL=small_cnn
#MODEL_NAME=small_cnn_data_aug
#MODEL=allconv
#MODEL_NAME=allconv_data_aug
#MODEL=resnet
#MODEL_NAME=ResNet50
#MODEL=vgg
#MODEL_NAME=VGG16
#MODEL=inception
#MODEL_NAME=InceptionV3
#MODEL=mobilenet
#MODEL_NAME=MobileNet
# MODEL=densenet
#MODEL_NAME=DenseNet
# MODEL=plain_grow
##MODEL_NAME=Plain_DataAug
#MODEL_NAME=Plain_pred_uncertainty
#MODEL=resnet_grow
MODEL=efficient_grow
#MODEL_NAME=ResNetGrow_data_aug_cifar100
#MODEL_NAME=${DATASET}_Aug_Margin_${MODEL}_C${CELL}_L${LAYER}_K${KERNEL}_B${BATCHSIZE}
#MODEL=autokeras
#MODEL_NAME=AutoKeras
#MODEL_NAME=${DATASET}_Aug${AUG}_ACC${ACCThresh}_${SAMPLE}_${MODEL}_C${CELL}_L${LAYER}_K${KERNEL}_B${BATCHSIZE}


# Growth Model
python3 run_experiment.py \
    --dataset=${DATASET} \
    --dataset_dir=${DATASET_DIR} \
    --score_model=${MODEL} \
    --standardize_data=False \
    --train_horizon=${HORIZON} \
    --trials=${TRIAL} \
    --sampling_method=${SAMPLE} \
    --labeling_sampling_method=${LABELING_SAMPLE} \
    --data_dir=../dataset/tf-data \
    --gpu=${GPU} \
    --cell=${CELL} \
    --layer=${LAYER} \
    --kernel=${KERNEL} \
    --warmstart_size=${WARMSTARTSIZE} \
    --batch_size=${BATCHSIZE}\
    --augmentation=${AUG}\
    --accthresh=${ACCThresh}\
    --minibatch=${MINIBATCH} \
    --profile=True \
    --test=True
#    --test=False
#    > log_${DATASET}_AUG${AUG}_ACC${ACCThresh}_T${SAMPLE}_L${LABELING_SAMPLE}_${MODEL}_C${CELL}_L${LAYER}_K${KERNEL}_B${BATCHSIZE}_WB${WARMSTARTSIZE}_MINIB${MINIBATCH}.txt

#
## Run evaluation.
#python3 utils/chart_data.py \
#    --dataset=${DATASET} \
#    --standardize=False \
#    --scoring_methods=${MODEL} \
#    --sampling_methods=margin,uniform \
#    --source_dir=./results/${MODEL_NAME} \
#    --save_dir=./results/${MODEL_NAME}
