import json
import os
from tqdm import tqdm
import numpy as np
import cv2

def check_modality():
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


def check_objects():
    root_directory = '/media/zhihao/Chi_SamSungT7/IKEA_ASM'
    text_path = '/media/zhihao/Chi_SamSungT7/IKEA_ASM/missing_obj.txt'

    env_lists = ['Kallax_Shelf_Drawer', 'Lack_Coffee_Table', 'Lack_Side_Table', 'Lack_TV_Bench']
    dev = 'dev3'
    for env in env_lists:
        item_lists = os.listdir(os.path.join(root_directory, env))
        for item in tqdm(item_lists):
            if item == 'special_test':
                continue
            item_dir = os.path.join(root_directory, env, item, 'dev3')
            video_path = os.path.join(item_dir, 'images', 'scan_video.avi')
            video_capture = cv2.VideoCapture(video_path)

            # Find the number of frames in the video
            num_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            video_capture.release()
            print("##########################################################")
            print("Video", video_path)
            print("Number of frames:", num_frames)
            for i in range(num_frames):
                obj_path = os.path.join(item_dir, 'seg', str(i)+'.json')
                # If not exists, write down the path
                if not os.path.exists(obj_path):
                    with open(text_path, 'a') as f:
                    # Write the text to the file
                        f.write(obj_path + '\n')
            # Release the video capture object
            

if __name__ == '__main__':
    check_objects()