import json
import os
from tqdm import tqdm
import numpy as np


root_directory = '/media/zhihao/Chi_SamSungT7/IKEA_ASM'


env_lists = ['Kallax_Shelf_Drawer', 'Lack_Coffee_Table', 'Lack_Side_Table', 'Lack_TV_Bench']
dev = 'dev3'
check_lists = ['depth', 'images', 'pose2d', 'pose3d', 'predictions', 'seg']

for env in env_lists:
    item_lists = os.listdir(os.path.join(root_directory, env))
    for item in tqdm(item_lists):
        item_directory = os.path.join(root_directory, env, item, 'dev3')
        data_types = os.listdir(item_directory)
        for check in check_lists:
            if check not in data_types:
                print("{} is missing under directory:{}".format(check, item_directory))
print("Checking data complete")