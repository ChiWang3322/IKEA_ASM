# STGCN
path1="runs/STGCN_50_wo_obj_split"
path2="runs/STGCN_50_w_obj_split"
# EGCN
path3="runs/EGCN_50_wo_obj_split"
path4="runs/EGCN_50_w_obj_split"

# AGCN
path5="runs/AGCN_50_wo_obj_split"
path6="runs/AGCN_50_w_obj_split"

# DSTA
path7="runs/DSTA_50_wo_obj_split"
path8="runs/DSTA_50_w_obj_split"





tensorboard --logdir $path1 --port 6007
tensorboard --logdir $path2 --port 6008
tensorboard --logdir $path3 --port 6009
tensorboard --logdir $path4 --port 6010
tensorboard --logdir $path5 --port 6011
tensorboard --logdir $path6 --port 6012
tensorboard --logdir $path7 --port 6013
tensorboard --logdir $path8 --port 6014

# python3 test_v3.py --config $CONFIG5
# python3 test_v3.py --config $CONFIG6

# python3 test_v3.py --config $CONFIG7
# python3 test_v3.py --config $CONFIG8

# python3 demo.py --config $DEMO
#python3 test.py --dataset_path $DATASET_PATH --pose_relative_path $POSE_REL_PATH --batch_size $BATCH_SIZE --frames_per_clip $FRAMES_PER_CLIP --arch $ARCH --model_path $LOGDIR --model 'best_classifier.pth'
