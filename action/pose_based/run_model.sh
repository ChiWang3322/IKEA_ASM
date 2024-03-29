# STGCN
CONFIG1="train_STGCN_wo_obj"
CONFIG2="train_STGCN_w_obj"
CONFIG1_SMALL="small_STGCN_wo_obj"
CONFIG2_SMALL="small_STGCN_w_obj"
# EGCN
CONFIG3="train_EGCN_wo_obj"
CONFIG4="train_EGCN_w_obj"
CONFIG3_SMALL="small_EGCN_wo_obj"
CONFIG4_SMALL="small_EGCN_w_obj"

# AGCN
CONFIG5="train_AGCN_wo_obj"
CONFIG6="train_AGCN_w_obj"
CONFIG5_SMALL="small_AGCN_wo_obj"
CONFIG6_SMALL="small_AGCN_w_obj"

# DSTA
CONFIG7="train_DSTA_wo_obj"
CONFIG8="train_DSTA_w_obj"
CONFIG7_SMALL="small_DSTA_wo_obj"
CONFIG8_SMALL="small_DSTA_w_obj"

# # Test small
# # STGCN
# python3 train_v4_split.py --config $CONFIG1_SMALL
# python3 train_v4_split.py --config $CONFIG2_SMALL
# # EGCN
# python3 train_v4_split.py --config $CONFIG3_SMALL
# python3 train_v4_split.py --config $CONFIG4_SMALL

# # AGCN
# python3 train_v4_split.py --config $CONFIG5_SMALL
# python3 train_v4_split.py --config $CONFIG6_SMALL
# # DSTA
# python3 train_v4_split.py --config $CONFIG7_SMALL
# python3 train_v4_split.py --config $CONFIG8_SMALL



# # Train
# # STGCN
# python3 train_v4_split.py --config $CONFIG1
# python3 train_v4_split.py --config $CONFIG2
# # EGCN
# python3 train_v4_split.py --config $CONFIG3
# python3 train_v4_split.py --config $CONFIG4

# AGCN
# python3 train_v4_split.py --config $CONFIG5
# python3 train_v4_split.py --config $CONFIG6
# # DSTA
# python3 train_v4_split.py --config $CONFIG7
# python3 train_v4_split.py --config $CONFIG8

# Train Customized A
# python3 train_v5_A.py --config train_EGCN_w_obj_A
# python3 test_v3.py --config train_EGCN_w_obj_A

# python3 train_v5_A.py --config train_EGCN_w_obj_A_ooh
# python3 test_v3.py --config train_EGCN_w_obj_A_ooh

# python3 train_v5_A.py --config train_EGCN_w_obj_A_oj
python3 train_v5_A_kit.py --config train_EGCN_w_obj_A_ooh_kit
# python3 train_v5_A.py --config train_STGCN_w_obj_A
# python3 train_v5_A.py --config train_AGCN_w_obj_A