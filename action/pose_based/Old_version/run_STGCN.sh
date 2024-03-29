#!/bin/bash

DATASET_PATH="/home/chiwang/Python/IKEA_Benchmark/IKEA_ASM_Dataset/dataset/ANU_ikea_dataset/"
POSE_REL_PATH="predictions/pose2d/openpose"
GPU_IDX=0
BATCH_SIZE=64
N_EPOCHS=500
FRAMES_PER_CLIP=64
ARCH='STGCN'
LOGDIR="./log/${ARCH}_${FRAMES_PER_CLIP}/"

python3 train.py --dataset_path $DATASET_PATH --pose_relative_path $POSE_REL_PATH --batch_size $BATCH_SIZE --n_epochs $N_EPOCHS --frames_per_clip $FRAMES_PER_CLIP --arch $ARCH --logdir $LOGDIR --gpu_idx $GPU_IDX #--refine --refine_epoch 2300

python3 test.py --dataset_path $DATASET_PATH --pose_relative_path $POSE_REL_PATH --batch_size $BATCH_SIZE --frames_per_clip $FRAMES_PER_CLIP --arch $ARCH --model_path $LOGDIR --model 'best_classifier.pth'