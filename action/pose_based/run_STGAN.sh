#!/bin/bash
# Data path and model architecture
DATASET_PATH="/home/chiwang/Python/IKEA_Benchmark/IKEA_ASM_Dataset/dataset/ANU_ikea_dataset/"
POSE_REL_PATH="predictions/pose2d/openpose"
ARCH='STGAN'
FRAMES_PER_CLIP=100
LOGDIR="./log/${ARCH}_${FRAMES_PER_CLIP}/"

# Training parameters
GPU_IDX=0
BATCH_SIZE=32
N_EPOCHS=500

LR=0.01
weight_decay=0.0005
LRSCHED=0
MOMENTUM=0

python3 train.py --dataset_path $DATASET_PATH --pose_relative_path $POSE_REL_PATH --batch_size $BATCH_SIZE --n_epochs $N_EPOCHS --frames_per_clip $FRAMES_PER_CLIP --arch $ARCH --logdir $LOGDIR --gpu_idx $GPU_IDX --lr $LR --momentum $MOMENTUM --weight_decay $weight_decay #--refine --refine_epoch 2300

python3 test.py --dataset_path $DATASET_PATH --pose_relative_path $POSE_REL_PATH --batch_size $BATCH_SIZE --frames_per_clip $FRAMES_PER_CLIP --arch $ARCH --model_path $LOGDIR --model 'best_classifier.pth'
