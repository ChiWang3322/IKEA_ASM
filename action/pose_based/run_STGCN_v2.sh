# CONFIG="test_small"
CONFIG1="train_STGCN_wo_obj"
CONFIG2="train_STGCN_w_obj"

# python3 train_v2.py --config $CONFIG3
python3 train_STGCN_v1.py --config $CONFIG2
#python3 test.py --dataset_path $DATASET_PATH --pose_relative_path $POSE_REL_PATH --batch_size $BATCH_SIZE --frames_per_clip $FRAMES_PER_CLIP --arch $ARCH --model_path $LOGDIR --model 'best_classifier.pth'
