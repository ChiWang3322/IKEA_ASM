import sqlite3
import os, sys, time, json
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2
import torch
import ujson, json
import pickle, logging, numpy as np
import pandas as pd
sys.path.append('../') # for dataset
class KITActionDataset():
    """
    KIT Action Dataset class
    """
    def __init__(self, dataset_path='/media/zhihao/Chi_SamSungT7/KIT_Bimanual', gt_filename='bimacs_rgbd_data_ground_truth', train_filename='train_cross_subject.txt',
                 test_filename='test_cross_subject.txt', dataset='train', hand='right_hand', action_list='action_list.json'):

        self.dataset_path = dataset_path
        self.gt_filename = gt_filename
        self.hand = hand
        self.train_path = os.path.join(dataset_path, train_filename)
        self.test_path = os.path.join(dataset_path, test_filename)
        self.data_path = self.train_path if dataset == 'train' else self.test_path
        self.action_list_path = os.path.join(self.dataset_path, action_list)
        with open(self.action_list_path, 'r') as file:
            self.action_list = json.load(file)
        self.num_classes = 14
        self.video_set = self.get_video_set()
    
    def get_video_set(self):
        video_set = []
        
        with open(self.data_path, 'r') as file:
            video_list = file.readlines()
            # print(type(train_file[0].rstrip('\n')))
            video_list = [x.rstrip('\n') for x in video_list]
        # Save video path
        for fname in video_list:
            # To take
            scan_path = os.path.join(self.dataset_path, 'images', fname, 'rgb')
            csv_file = os.path.join(scan_path, 'metadata.csv')
            metadata = pd.read_csv(csv_file)
            metadata = metadata.set_index('name')
            num_frame = metadata.loc['frameCount',:]['value']
            
            # Save corresponding video labels
            labels = self.get_labels(fname)
            
            # print('label size:', labels.shape)
            # print('scan_path:', scan_path)
            # print('num frame:', num_frame)

            # Append to video set
            video_set.append((scan_path, labels, num_frame))
        return video_set

    def get_labels(self, filename):
        """
        Convert KIT Bimanual action label to one hot encoding label

        Parameters
        ----------
        filename : take_0.json file name (full path)
        Returns
        -------
        labels: one_hot frame labels (allows multi-label)
        """
        scan_path = os.path.join(self.dataset_path, 'labels', filename+'.json')
        csv_file = os.path.join(self.dataset_path, 'images', filename, 'rgb', 'metadata.csv')
        metadata = pd.read_csv(csv_file)
        metadata = metadata.set_index('name')
        num_frame = int(metadata.loc['frameCount',:]['value'])
        # read json file
        with open(scan_path, 'r') as file:
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
        # if num_frame != labels.shape[1]:
        #     print('----------')
        #     print('scan_path:', scan_path)
        #     print('num frame:', num_frame)
        #     print('label size:', labels.shape)
        # print('----------')
        # print('num frame inside function:', num_frame)
        # print('label size inside function:', labels.shape)
        return labels


# This dataset loads videos (not single frames) and cuts them into clips
class KITActionVideoClipDataset(KITActionDataset):
    def __init__(self, dataset_path='/media/zhihao/Chi_SamSungT7/KIT_Bimanual', gt_filename='bimacs_rgbd_data_ground_truth', train_filename='train_cross_subject.txt',
                 test_filename='test_cross_subject.txt', dataset='train', hand='right_hand', action_list='action_list.json',
                 frames_per_clip=50, frame_skip=1):
        super().__init__(dataset_path=dataset_path, gt_filename=gt_filename, train_filename=train_filename,
                 test_filename=test_filename, dataset=dataset, hand=hand, action_list=action_list)
        self.frame_skip = frame_skip
        self.frames_per_clip = frames_per_clip
        self.clip_set = self.get_clips()


    def get_clips(self):
        # extract equal length video clip segments from the full video dataset
        clip_dataset = []
        label_count = np.zeros(self.num_classes)
        for i, data in enumerate(self.video_set):
            n_frames = int(data[2])
            n_clips = int(n_frames / (self.frames_per_clip * self.frame_skip))
            remaining_frames = n_frames % (self.frames_per_clip * self.frame_skip)
            for j in range(0, n_clips):
                for k in range(0, self.frame_skip):
                    start = j * self.frames_per_clip * self.frame_skip + k
                    end = (j + 1) * self.frames_per_clip * self.frame_skip
                    label = data[1][:, start:end:self.frame_skip]

                    frame_ind = np.arange(start, end, self.frame_skip).tolist()
                    clip_dataset.append((data[0], label, frame_ind, self.frames_per_clip, i, 0))
            # Process the last clip
            if not remaining_frames == 0:
                # print("remaining frames:", remaining_frames)
                frame_pad =  self.frames_per_clip - remaining_frames
                start = n_clips * self.frames_per_clip * self.frame_skip + k
                end = start + remaining_frames
                new_start = start - frame_pad
                label = data[1][:, new_start:end]
                x, y = label.shape
                if y == 49:
                    # print('---------------------------------------')
                    # print('data[1] shape:', data[1].shape)
                    # print('num frame:', data[2])
                    # print('Path:', data[0])
                    
                    # print('shape:', label.shape)
                    # print('frame pad:', frame_pad)
                    # print("Remain:", remaining_frames)
                    # print("start:", start)
                    # print('Start - frame_pad:', start - frame_pad)

                    # print("End:", end)
                    # print(np.arange(start-frame_pad, end, self.frame_skip).tolist())
                    pass
                frame_ind = np.arange(start-frame_pad, end, self.frame_skip).tolist()
                clip_dataset.append((data[0], label, frame_ind, self.frames_per_clip, i, frame_pad))
        return clip_dataset

    

    def __len__(self):
        # 'Denotes the total number of samples'
        return len(self.clip_set)

    def __getitem__(self, index):
        # 'Generate one sample of data'
        video_full_path, labels, frame_ind, n_frames_per_clip, vid_idx, frame_pad = self.clip_set[index]

        imgs = self.load_rgb_frames(video_full_path, frame_ind)
        imgs = self.transform(imgs)

        return self.video_to_tensor(imgs), torch.from_numpy(labels), vid_idx, frame_pad


# This dataset loads per frame poses
class KITPoseActionVideoClipDataset(KITActionVideoClipDataset):
    def __init__(self, dataset_path='/media/zhihao/Chi_SamSungT7/KIT_Bimanual', gt_filename='bimacs_rgbd_data_ground_truth', 
                 train_filename='train_cross_subject.txt', test_filename='test_cross_subject.txt', dataset='train', 
                 hand='right_hand', action_list='action_list.json', frames_per_clip=50, frame_skip=1, 
                 pose_path='body_pose',obj_path='2d_object', with_obj=True, obj_dim=3):
        super().__init__(dataset_path=dataset_path, gt_filename=gt_filename, 
                        train_filename=train_filename, test_filename=test_filename, dataset=dataset, 
                        hand=hand, action_list=action_list, frames_per_clip=frames_per_clip, frame_skip=frame_skip)
        self.pose_path = pose_path
        self.frames_per_clip = frames_per_clip
        self.frame_skip = frame_skip
        self.obj_path = obj_path
        self.with_obj = with_obj
        self.obj_dim = obj_dim
        self.Width = 640
        self.Height = 480
    def load_poses(self, video_full_path, frame_ind):
        # print(video_full_path)
        pose_seq = []
        # print(frame_ind)
        video_full_path = video_full_path.replace('rgb', self.pose_path)
        # print("Pose path:",video_full_path)
        def reorder_skeleton(pose_json_filename):
            with open(pose_json_filename) as json_file:
                data = json.load(json_file)
                data = data[0]
                keys = list(data.keys())
                # print(keys)
            pose = []
            for i in range(len(keys)):
                key = keys[i]

                if key[0] == 'L':
                    joint = [0, 0]
                else:
                    joint = [data[key]['x'], data[key]['y']]
                # if joint[0] == 0.0 or joint[1] == 0.0:
                #     print(pose_json_filename)

                # print('joint:', joint)
                pose.append(joint)
            return pose
        for i in frame_ind:
            # Read json file for each frame
            pose_json_filename = os.path.join(video_full_path, 'frame_' + str(i) + '.json')
            # data = utils.read_pose_json(pose_json_filename)
            pose = reorder_skeleton(pose_json_filename) # x, y
            # print(pose)
            

            pose_seq.append(pose)
        pose_seq = np.array(pose_seq)
        pose_seq = torch.tensor(pose_seq, dtype=torch.float32)
        # print('STGCN pose_seq pose.shape:', pose_seq.size())
        pose_seq = pose_seq[:, :, 0:2].unsqueeze(-1)  # format: frames, joints, coordinates, N_people
        # print('STGCN pose_seq pose.shape:', pose_seq.size())
        pose_seq = pose_seq.permute(2, 0, 1, 3)  # format: coordinates, frames, joints, N_people


        return pose_seq
    def load_obj(self, video_full_path, frame_ind, pose_size):
        sample_path = video_full_path.replace('rgb', self.obj_path)
        c, t, v, m = pose_size
        # Here define how many object data you want
        object_data = np.zeros([self.obj_dim, t, v, m])
        if self.with_obj:
            # print("yes")
            for count, index in enumerate(frame_ind):
                
                object_full_path = os.path.join(sample_path, 'frame_'+str(index)+'.json')
                if not os.path.exists(object_full_path):
                    
                    print("No objects detected in this frame")
                    # object_full_path = os.path.join(sample_path, str(index)+'.json')
                    continue
                with open(object_full_path) as json_file:
                    # try:
                    objects = ujson.load(json_file)
                    json_file.close()
                    # except:
                    #     print(object_full_path)

                for i, object_2d in enumerate(objects): 
                    # C T V M

                    box_width = object_2d['w'] * self.Width 
                    box_high = object_2d['h'] * self.Height

                    center_x = object_2d['x'] * self.Width 
                    center_y = object_2d['y'] * self.Height

                    # left_top_x = object_2d[0]
                    # left_top_y = object_2d[1]

                    # right_bottom_x = object_2d[0] + box_width
                    # right_bottom_y = object_2d[1] + box_high

                    # left_bottom_x = object_2d[0]
                    # left_bottom_y = object_2d[1] + box_high

                    # right_top_x = object_2d[0] + box_width
                    # right_top_y = object_2d[1]
                    # (center, left_top, right_bottom, left_bottom, right_top)
                    object_data[:2, count, i , 0] = [center_x, center_y]
                    # object_data[:2, count, i , 0] = [center_x, center_y]

                    # object_data[:4, count, i , 0] = object_tmp['bbox']
                    object_data[-1, count, i, 0] = object_2d['class_id']

        else:
            object_data = 0
        object_data = torch.tensor(object_data, dtype=torch.float32)
        # print(object_data[:, 1, 0, 0])
        # print(object_data.size())
        # print("Load object data cost time:", time.time()-start)
        return object_data

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


    def __getitem__(self, index):
        # 'Generate one sample of data'
        video_full_path, labels, frame_ind, n_frames_per_clip, vid_idx, frame_pad = self.clip_set[index]
        # print(frame_ind)
        
        poses = self.load_poses(video_full_path, frame_ind)
        object_data = self.load_obj(video_full_path, frame_ind, poses.size())

        return poses, torch.from_numpy(labels), vid_idx, frame_pad, object_data, video_full_path, frame_ind




if __name__ == "__main__":

    # print(dataset.action_list)
    dataset = KITPoseActionVideoClipDataset()
    for i in range(50):
        pose, labels, vid_idx, frame_pad, object_data, video_path = dataset[i]
        print(pose.size())
        print(labels.size())
        print(object_data.size())
        print('------------------------')
    # print(object_data[:, 1, :, 0])
    

