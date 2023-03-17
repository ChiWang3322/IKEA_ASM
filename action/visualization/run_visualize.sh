# CONFIG="test_small"

DBFILE='ikea_annotation_db_full'
DATASET_PATH='/home/zhihao/ChiWang_MA/IKEA_ASM/dataset/'
POSE_PATH='predictions/pose2d/openpose'
# python3 train_v2.py --config $CONFIG3
python3 pose_visualization.py --db_filename $DBFILE --dataset_path $DATASET_PATH --pose_path $POSE_PATH
#python3 test.py --dataset_path $DATASET_PATH --pose_relative_path $POSE_REL_PATH --batch_size $BATCH_SIZE --frames_per_clip $FRAMES_PER_CLIP --arch $ARCH --model_path $LOGDIR --model 'best_classifier.pth'
