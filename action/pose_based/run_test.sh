# STGCN
CONFIG1="train_STGCN_wo_obj"
CONFIG2="train_STGCN_w_obj"
CONFIG2_A="train_STGCN_w_obj_A"
CONFIG1_SMALL="small_STGCN_wo_obj"
CONFIG2_SMALL="small_STGCN_w_obj"
# EGCN
CONFIG3="train_EGCN_wo_obj"
CONFIG4="train_EGCN_w_obj"
CONFIG4_A="train_EGCN_w_obj_A"
CONFIG3_SMALL="small_EGCN_wo_obj"
CONFIG4_SMALL="small_EGCN_w_obj"

# AGCN
CONFIG5="train_AGCN_wo_obj"
CONFIG6="train_AGCN_w_obj"
CONFIG6_A="train_AGCN_w_obj_A"
CONFIG5_SMALL="small_AGCN_wo_obj"
CONFIG6_SMALL="small_AGCN_w_obj"

# DSTA
CONFIG7="train_DSTA_wo_obj"
CONFIG8="train_DSTA_w_obj"
CONFIG7_SMALL="small_DSTA_wo_obj"
CONFIG8_SMALL="small_DSTA_w_obj"

DEMO="demo"


# python3 test_v3.py --config $CONFIG1
# python3 test_v3.py --config $CONFIG2

# python3 test_v3.py --config $CONFIG3
# python3 test_v3.py --config $CONFIG4

# python3 test_v3.py --config $CONFIG5
# python3 test_v3.py --config $CONFIG6

# python3 test_v3.py --config $CONFIG7
# python3 test_v3.py --config $CONFIG8

# python3 test_v3.py --config train_EGCN_w_obj_A
# python3 test_v3.py --config train_EGCN_w_obj_A_ooh
python3 test_v3.py --config train_EGCN_w_obj_A_oj
# python3 test_v3.py --config $CONFIG4_A
# python3 test_v3.py --config $CONFIG6_A