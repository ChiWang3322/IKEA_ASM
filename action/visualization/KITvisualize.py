import os
import shutil
import cv2
import vis_utils
import numpy as np
import pandas as pd
import json
import random
import math
import sys
sys.path.append('../')
import utils
import sqlite3
#TODO implement adjust depth to rgb image size
from KITActionDataset import KITPoseActionVideoClipDataset as Dataset



class FunctionSwitcher:

    def __init__(self, scan_path, dataset_path='/media/zhihao/Chi_SamSungT7/KIT_Bimanual', 
                dataset='train', hand='right_hand'):
        self.dataset_path = dataset_path
        self.hand = hand
        self.scan_path = scan_path
        self.video_path = os.path.join(dataset_path, 'images', self.scan_path, 'rgb')
        with open('/media/zhihao/Chi_SamSungT7/KIT_Bimanual/action_list.json', 'r') as file:
            self.action_list = json.load(file)
        
        self.num_classes = 14
        self.labels = self.get_labels()
    def get_labels(self):
        """
        Convert KIT Bimanual action label to one hot encoding label

        Parameters
        ----------
        filename : take_0.json file name (full path)
        Returns
        -------
        labels: one_hot frame labels (allows multi-label)
        """
        label_path = os.path.join(self.dataset_path, 'labels', self.scan_path+'.json')
        csv_file = os.path.join(self.dataset_path, 'images', self.scan_path, 'rgb', 'metadata.csv')
        metadata = pd.read_csv(csv_file)
        metadata = metadata.set_index('name')
        num_frame = int(metadata.loc['frameCount',:]['value'])
        # read json file
        with open(label_path, 'r') as file:
            json_data = json.load(file)
            data = json_data[self.hand]
            label_end = data[-1]
        # Transform label to one hot encoding
        labels = np.zeros((self.num_classes, num_frame), np.float32)
        value = data[0]
        index = 0
        while value < label_end:
            start = value
            category = data[index + 1]
            end = data[index + 2]
            labels[category, start:end] = 1
            # print("Cat:", category)
            # print("labels:", labels[:, start:end])
            value = end
            index += 2
        return labels

    def get_list_from_file(self, filename):
        """
        retrieve a list of lines from a .txt file
        :param :
        :return: list of atomic actions
        """
        with open(filename) as f:
            line_list = f.read().splitlines()
        # line_list.sort()
        return line_list

    
    


    def change_video(self, video_path):
        self.scan_path = video_path
    def get_obj(self, obj_json_filename, Height, Width):
        with open(obj_json_filename) as json_file:
            data = json.load(json_file)
        obj_data = []
        for bbox in data:
            h, w, x, y = bbox['h'] * Height, bbox['w'] * Width, bbox['x'] * Width, bbox['y'] * Height
            class_id = bbox['class_id']
            class_color = bbox['class_color']
            obj_data.append((h, w, x, y, class_id, class_color))


        return obj_data
    def display(self):
        # To rgb directory
        video_path = self.video_path

        # scan_path = os.path.join(self.dataset_path, 'images', fname)
        csv_file = os.path.join(video_path, 'metadata.csv')
        metadata = pd.read_csv(csv_file)
        metadata = metadata.set_index('name')
        num_frame = int(metadata.loc['frameCount',:]['value'])
        Height = float(metadata.loc['frameHeight',:]['value'])
        Width = float(metadata.loc['frameWidth',:]['value'])
        labels = self.get_labels()
        labels = np.argmax(labels, axis=0)
        frame_index = 0    

        # Read until video is completed
        while frame_index < num_frame:
            # Read frame
            frame_path = os.path.join(video_path, 'frame_' + str(frame_index) + '.png')
            frame = cv2.imread(frame_path)
            # Get joint data
            joint_data = []
            pose_json_filename = video_path.replace('rgb', 'body_pose')
            pose_json_filename = os.path.join(pose_json_filename, 'frame_' + str(frame_index) + '.json')
            with open(pose_json_filename) as json_file:
                pose_data = json.load(json_file)
                pose_data = pose_data[0]
            keys = list(pose_data.keys())
            # tmp = []
            for i in range(len(keys)):
                # if key[0] == 'L':
                #     joint = (0, 0) 
                key = keys[i]
                # else:
                joint = (pose_data[key]['x'], pose_data[key]['y'])
                # tmp.append(key)
                joint_data.append(joint)
            # Get object data
            obj_json_filename = video_path.replace('rgb', '2d_object')
            obj_json_filename = os.path.join(obj_json_filename, 'frame_' + str(frame_index) + '.json')
            obj_data = self.get_obj(obj_json_filename, Height, Width)

            for i, obj in enumerate(obj_data):
                bbox = obj_data[i]

                # bbox = obj['bbox'] # <bb_left>, <bb_top>, <bb_width>, <bb_height>
                h, w, x, y, class_id, class_color = bbox[0], bbox[1], bbox[2], bbox[3], bbox[4], bbox[5]
                center_x = int(x)
                center_y = int(y)
                pt1 = (int(x - w / 2), int(y - h / 2))
                pt2 = (int(x + w / 2), int(y + h / 2))
                cv2.rectangle(frame, pt1, pt2, class_color, thickness=2)
                cv2.circle(frame, (center_x, center_y), 2, (255, 0, 0), thickness=-1, lineType=cv2.FILLED)

            
            # Define the text to be written
            txt = 'Action: ' + str(self.action_list[str(labels[frame_index])]) + '(' + self.hand + ')' 
            # Define the font
            font = cv2.FONT_HERSHEY_SIMPLEX
            # Define the position of the text
            position = (50, 50)
            # Define the font scale
            fontScale = 1
            # Define the color of the text
            color = (0, 0, 255)
            # Define the thickness of the text
            thickness = 2
            # Write the text on the image
            cv2.putText(frame, txt, position, font, fontScale, (255,0,0), thickness)
            for i, point in enumerate(joint_data):
                cv2.circle(frame, (int(point[0]), int(point[1])), 1, (0, 165, 255), thickness=-1, lineType=cv2.FILLED)
                cv2.putText(frame, keys[i], (int(point[0]), int(point[1])), font, 0.5, (0,0,255), 1)
                        
            cv2.imshow('KITAction', frame)
            frame_index += 1
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
    def display_rgb(self):
        # To rgb directory
        video_path = self.video_path

        # scan_path = os.path.join(self.dataset_path, 'images', fname)
        csv_file = os.path.join(video_path, 'metadata.csv')
        metadata = pd.read_csv(csv_file)
        metadata = metadata.set_index('name')
        num_frame = int(metadata.loc['frameCount',:]['value'])
        labels = self.get_labels()
        labels = np.argmax(labels, axis=0)
        frame_index = 0        
        # Read until video is completed
        while frame_index < num_frame:
            


            frame_path = os.path.join(video_path, 'frame_' + str(frame_index) + '.png')
            frame = cv2.imread(frame_path)
            # Define the text to be written
            txt = 'Action: ' + str(self.action_list[str(labels[frame_index])]) + '(' + self.hand + ')' 
            # Define the font
            font = cv2.FONT_HERSHEY_SIMPLEX
            # Define the position of the text
            position = (50, 50)
            # Define the font scale
            fontScale = 1
            # Define the color of the text
            color = (0, 0, 255)
            # Define the thickness of the text
            thickness = 2
            # Write the text on the image
            cv2.putText(frame, txt, position, font, fontScale, (255,0,0), thickness)
            cv2.imshow('KITAction', frame)
            frame_index += 1
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break

    def display_depth(self):
        video_path = os.path.join(self.scan_path, self.modality, 'depth', 'scan_video.avi')
        cap = cv2.VideoCapture(video_path)
        self.get_label()
        # Check if camera opened successfully
        if (cap.isOpened()== False): 
            print("Error opening video stream or file")
        
        # Read until video is completed
        while(cap.isOpened()):
            ret, img = cap.read()
            if ret:
                cv2.imshow('Frame',img)
            
                # Press Q on keyboard to  exit
                if cv2.waitKey(60) == ord('q'):
                    break
            else:
                break
    
        
    def find_id(self, image_name, test_data):
        pass
        # for item in test_data['images']:
        #     if item['file_name'].find(image_name) != -1:
        #         return item['id']
        # return -1
    
    def get_active_person(self, people, center=(960, 540), min_bbox_area=20000):
        """
           Select the active skeleton in the scene by applying a heuristic of findng the closest one to the center of the frame
           then take it only if its bounding box is large enough - eliminates small bbox like kids
           Assumes 100 * 200 minimum size of bounding box to consider
           Parameters
           ----------
           data : pose data extracted from json file
           center: center of image (x, y)
           min_bbox_area: minimal bounding box area threshold

           Returns
           -------
           pose: skeleton of the active person in the scene (flattened)
           """

        pose = None
        min_dtc = float('inf')  # dtc = distance to center
        for person in people:
            current_pose = np.array(person['pose_keypoints_2d'])
            joints_2d = np.reshape(current_pose, (-1, 3))[:, :2]
            if 'boxes' in person.keys():
                # maskrcnn
                bbox = person['boxes']
            else:
                # openpose
                idx = np.where(joints_2d.any(axis=1))[0]
                bbox = [np.min(joints_2d[idx, 0]),
                        np.min(joints_2d[idx, 1]),
                        np.max(joints_2d[idx, 0]),
                        np.max(joints_2d[idx, 1])]

            A = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])  # bbox area
            bbox_center = (bbox[0] + (bbox[2] - bbox[0]) / 2, bbox[1] + (bbox[3] - bbox[1]) / 2)  # bbox center

            dtc = np.sqrt(np.sum((np.array(bbox_center) - np.array(center)) ** 2))
            if dtc < min_dtc:
                closest_pose = current_pose
                if A > min_bbox_area:
                    pose = closest_pose
                    min_dtc = dtc
        # if all bboxes are smaller than threshold, take the closest
        if pose is None:
            pose = closest_pose
        return pose

    def compute_skeleton_distance_to_center(self, skeleton, center=(960, 540)):
        """
        Compute the average distance between a given skeleton and the cetner of the image
        Parameters
        ----------
        skeleton : 2d skeleton joint poistiions
        center : image center point

        Returns
        -------
            distance: the average distance of all non-zero joints to the center
        """
        idx = np.where(skeleton.any(axis=1))[0]
        diff = skeleton - np.tile(center, [len(skeleton[idx]), 1])
        distances = np.sqrt(np.sum(diff ** 2, 1))
        mean_distance = np.mean(distances)

        return mean_distance
    
    def display_rgb_and_depth(self):


        # Define paths to RGB and depth videos
        rgb_video_path = os.path.join(self.scan_path, self.modality, 'images', 'scan_video.avi')
        depth_video_path = os.path.join(self.scan_path, self.modality, 'depth', 'scan_video.avi')

        # Create VideoCapture objects for both videos
        rgb_cap = cv2.VideoCapture(rgb_video_path)
        depth_cap = cv2.VideoCapture(depth_video_path)

        # Check if videos opened successfully
        if not (rgb_cap.isOpened() and depth_cap.isOpened()):
            print("Error opening videos")
            exit()

        # Get video dimensions
        # width = int(depth_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        # height = int(depth_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Create window to display both videos
        cv2.namedWindow("RGB and Depth Video", cv2.WINDOW_NORMAL)

        while True:
            # Read frames from both videos
            ret1, frame1 = rgb_cap.read()
            ret2, frame2 = depth_cap.read()

            # If either video has reached its end, break loop
            if not (ret1 and ret2):
                break

            # Combine RGB and depth frames horizontally
            frame1 = cv2.resize(frame1, (frame2.shape[1], frame2.shape[0]))
            combined_frame = cv2.hconcat([frame1, frame2])

            # Display combined frame
            cv2.imshow("RGB and Depth Video", combined_frame)

            # Press 'q' to exit
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

        # Release video capture objects and destroy window
        rgb_cap.release()
        depth_cap.release()
        cv2.destroyAllWindows()

    def get_pesudo_image(self):
        video_path = os.path.join(self.scan_path, self.modality, 'images', 'scan_video.avi')
        cap = cv2.VideoCapture(video_path)
        t_count = 50
        frame_ind = 0
        skeleton_array = []
        while(cap.isOpened()):
        # Capture frame-by-frame
            ret, img = cap.read()
            if ret == True:
                self.file_idx = frame_ind
                j2d, _ = self.pose()
                
                skeleton_array.append(j2d.tolist())
                frame_ind += 1
                if frame_ind >= t_count:
                    break
        return skeleton_array
        
def get_list_from_file(filename):
    """
    retrieve a list of lines from a .txt file
    :param :
    :return: list of atomic actions
    """
    with open(filename) as f:
        line_list = f.read().splitlines()
    # line_list.sort()
    return line_list

if __name__ == '__main__':
    switcher = FunctionSwitcher(scan_path='subject_3/task_1_k_cooking/take_5')

    switcher.display()
    # dataset_path='/media/zhihao/Chi_SamSungT7/KIT_Bimanual', dataset='train', hand='right_hand')

    # dataset = Dataset(dataset_path='/media/zhihao/Chi_SamSungT7/KIT_Bimanual', gt_filename='bimacs_rgbd_data_ground_truth', 
    #             train_filename='train_cross_subject.txt', test_filename='test_cross_subject.txt', dataset='train', 
    #             hand='right_hand', action_list='action_list.json', frames_per_clip=200, frame_skip=1, 
    #             pose_path='body_pose',obj_path='2d_object', with_obj=True, obj_dim=3)
    
    # pose, labels, vid_idx, frame_pad, object_data, video_path, frame_ind = dataset[33] # video_path to rgb
    # labels = labels.numpy()
    # labels = np.argmax(labels, axis = 0)
    # # print(labels)
    # pose = pose.numpy()
    # # print(pose.size())
    # csv_file = os.path.join(video_path, 'metadata.csv')
    # metadata = pd.read_csv(csv_file)
    # metadata = metadata.set_index('name')
    # num_frame = int(metadata.loc['frameCount',:]['value'])
    # for i in range(len(frame_ind)):
    #     frame_index = frame_ind[i]
    #     joint_data = pose[:, i, :, :]
    #     frame_path = os.path.join(video_path, 'frame_' + str(frame_index) + '.png')
    #     frame = cv2.imread(frame_path)
    #     # Define the text to be written
    #     txt = 'Action: ' + str(switcher.action_list[str(int(labels[i]))]) + '(' + switcher.hand + ')' 
    #     # Define the font
    #     font = cv2.FONT_HERSHEY_SIMPLEX
    #     # Define the position of the text
    #     position = (50, 50)
    #     # Define the font scale
    #     fontScale = 1
    #     # Define the color of the text
    #     color = (0, 0, 255)
    #     # Define the thickness of the text
    #     thickness = 2
    #     # Write the text on the image
    #     for j in range(25):
    #         point = joint_data[:, j]
    #         cv2.circle(frame, (int(point[0]), int(point[1])), 1, (0, 165, 255), thickness=-1, lineType=cv2.FILLED)
    #         # cv2.putText(frame, str(i), (int(point[0]), int(point[1])), font, 0.2, (0,0,255), 1)
    #     cv2.putText(frame, txt, position, font, fontScale, (255,0,0), thickness)
    #     cv2.imshow('KITAction', frame)
    #     # frame_index += 1
    #     if cv2.waitKey(70) & 0xFF == ord('q'):
    #         break
        
    
    
        



