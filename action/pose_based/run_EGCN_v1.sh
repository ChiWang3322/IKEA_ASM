#!/bin/bash
# Data path and model architecture
DATASET_PATH="/home/zhihao/ChiWang_MA/IKEA_ASM/dataset"
POSE_REL_PATH="predictions/pose2d/openpose"
ARCH='EGCN'
FRAMES_PER_CLIP=50
LOGDIR="./log/${ARCH}_${FRAMES_PER_CLIP}_w_ojb/"

# Training parameters
GPU_IDX=0
BATCH_SIZE=32
N_EPOCHS=100
LR=0.01
weight_decay=0.0001
#refine = 1
# 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
python3 train.py --dataset_path $DATASET_PATH --pose_relative_path $POSE_REL_PATH --batch_size $BATCH_SIZE --n_epochs $N_EPOCHS --frames_per_clip $FRAMES_PER_CLIP --arch $ARCH --logdir $LOGDIR --gpu_idx $GPU_IDX --lr $LR --weight_decay $weight_decay #--refine refine #--refine_epoch 2300

python3 test.py --dataset_path $DATASET_PATH --pose_relative_path $POSE_REL_PATH --batch_size $BATCH_SIZE --frames_per_clip $FRAMES_PER_CLIP --arch $ARCH --model_path $LOGDIR --model 'best_classifier.pth'