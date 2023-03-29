import json
import os
from tqdm import tqdm
import numpy as np
import cv2
import shutil


class DatasetChecker():
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.env_lists = env_lists = ['Kallax_Shelf_Drawer', 'Lack_Coffee_Table', 'Lack_Side_Table', 'Lack_TV_Bench']
        self.check_lists = check_lists 
        self.dev = 'dev3'

    def check_modality():
        # root_directory = '/media/zhihao/Chi_SamSungT7/IKEA_ASM'


        # env_lists = ['Kallax_Shelf_Drawer', 'Lack_Coffee_Table', 'Lack_Side_Table', 'Lack_TV_Bench']
        dev = 'dev3'
        # check_lists = ['depth', 'images', 'pose2d', 'pose3d', 'predictions', 'seg']

        for env in self.env_lists:
            item_lists = os.listdir(os.path.join(self.root_directory, env))
            for item in tqdm(item_lists):
                item_directory = os.path.join(self.root_directory, env, item, 'dev3')
                data_types = os.listdir(item_directory)
                for check in self.check_lists:
                    if check not in data_types:
                        print("{} is missing under directory:{}".format(check, item_directory))
        print("Checking data complete")


def check_objects():
    root_directory = '/media/zhihao/Chi_SamSungT7/IKEA_ASM'
    text_path = '/media/zhihao/Chi_SamSungT7/IKEA_ASM/missing_obj_now.txt'

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
            # Release the video capture object
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
    if not os.path.exists(text_path):
        print("No missing frames!")
def check_trans(step):
    root_directory = '/media/zhihao/Chi_SamSungT7/IKEA_ASM'
    text_path = '/media/zhihao/Chi_SamSungT7/IKEA_ASM/missing_trans_now.txt'

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
            # Release the video capture object
            video_capture.release()
            print("##########################################################")
            print("Video", video_path)
            print("Number of frames:", num_frames)
            for i in range(num_frames):
                obj_path = os.path.join(item_dir, 'trans_'+str(step), str(i)+'.json')
                # If not exists, write down the path
                if not os.path.exists(obj_path):
                    with open(text_path, 'a') as f:
                    # Write the text to the file
                        f.write(obj_path + '\n')
    if not os.path.exists(text_path):
        print("No missing frames!")
def fix_objects():
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
            # Release the video capture object
            video_capture.release()
            print("##########################################################")
            print("Video", video_path)
            print("Number of frames:", num_frames)
            for i in range(num_frames):
                obj_path = os.path.join(item_dir, 'seg', str(i)+'.json')
                # If not exists, write down the path
                if not os.path.exists(obj_path):
                    # specify the original file path and the new file path with new name
                    original_file_path = os.path.join(item_dir, 'seg', str(i-1)+'.json')
                    new_file_path = obj_path

                    # copy the original file to the new file path and rename it
                    shutil.copyfile(original_file_path, new_file_path)
            

if __name__ == '__main__':
    # fix_objects()
    check_objects()
    check_trans()