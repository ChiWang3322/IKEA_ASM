import os
import shutil
import cv2
import vis_utils
import numpy as np
import json
import random
import math
import sys
sys.path.append('../')
import utils
import sqlite3
#TODO implement adjust depth to rgb image size
class IntensityLengthError(Exception):
    pass



class FunctionSwitcher:

    def __init__(self, scan_path, output_path=None, modality='dev3', resize_factor=None, context = True, 
                dataset_dir = '/media/zhihao/Chi_SamSungT7/IKEA_ASM', db_filename='ikea_annotation_db_full',
                action_list_filename='atomic_action_list.txt', action_object_relation_filename='action_object_relation_list.txt',
                video_index=0, logdir = None):
        self.video_index = video_index
        self.scan_path = scan_path
        self.output_path = output_path
        self.logdir = logdir
        # os.makedirs(output_path, exist_ok=True)
        self.modality = modality
        self.device = modality
        # self.name_mapping_dict = name_mapping_dict
        self.resize_factor = resize_factor
        self.file_idx = 0
        self.context = context
        self.dataset_dir = dataset_dir
        # Connect database file
        self.db_file = os.path.join(dataset_dir, db_filename)
        self.db = sqlite3.connect(self.db_file)
        self.db.row_factory = sqlite3.Row
        self.cursor_vid = self.db.cursor()
        self.cursor_annotations = self.db.cursor()
        # Load action list file
        self.action_list_filename = os.path.join(dataset_dir, 'indexing_files', action_list_filename)
        self.action_object_relation_filename = os.path.join(dataset_dir, 'indexing_files', action_object_relation_filename)
        
        # load action list and object relation list and sort them together
        self.action_list = self.get_list_from_file(self.action_list_filename)
        self.ao_list = self.get_action_object_relation_list()

        self.ao_list = [x for _, x in sorted(zip(self.action_list, self.ao_list))]
        self.action_list.sort()
        self.action_list.insert(0, "NA")  #  0 label for unlabled frames
        # print(self.action_list)
        # print(self.ao_list)
        self.num_classes = len(self.action_list)
    def get_label(self):
        # print("This is a test function")
        # table = self.get_annotated_videos_table(device='dev3')
        # print(table)
        # for row in table:
        #     video_path = row['video_path']
        #     print(video_path)
        video_path = os.path.join(self.scan_path.split('/')[-2], self.scan_path.split('/')[-1], self.device, 'images')

        self.vid_id = self.get_video_id_from_video_path(video_path=video_path)
        # print(vid_id)
        
        annotation_table = self.get_video_annotations_table(video_idx=self.vid_id).fetchall()
        video_info = self.get_video_table(self.vid_id).fetchone()
        n_frames = video_info["nframes"]
        # print(self.num_classes)
        label = np.zeros((self.num_classes, n_frames), np.float32) # allow multi-class representation
            
        label[0, :] = np.ones((1, n_frames), np.float32)   # initialize all frames as background|transition
        # print(n_frames)
        for ann_row in annotation_table:
            # print(ann_row)
            # print('id:',ann_row['id'])
            # print('video_id:',ann_row['video_id'])
            # print('atomic_action_id:',ann_row['atomic_action_id'])
            # print('action_counter:',ann_row['action_counter'])
            # print('description:',ann_row['action_description'])
            # print('object_id:',ann_row['object_id'])
            # print('object_name:',ann_row['object_name'])
            # print('start:',ann_row['starting_frame'])
            # print('ending:',ann_row['ending_frame'])
            # print('-----------------------------')
            atomic_action_id = ann_row["atomic_action_id"]  # map the labels
            object_id = ann_row["object_id"]
            action_id = self.get_action_id(atomic_action_id, object_id)
            end_frame = ann_row['ending_frame'] if ann_row['ending_frame'] < n_frames else n_frames
            if action_id is not None:
                label[:, ann_row['starting_frame']:end_frame] = 0  # remove the background label
                label[action_id, ann_row['starting_frame']:end_frame] = 1
        self.label = np.argmax(label, axis = 0)


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
    def get_action_object_relation_list(self):
        # read the action-object relation list file and return a list of integers
        with open(self.action_object_relation_filename, 'r') as file:
            a_o_relation_list = file.readlines()
            a_o_relation_list = [x.rstrip('\n').split() for x in a_o_relation_list]
            a_o_relation_list = [[int(y) for y in x] for x in a_o_relation_list]
        return a_o_relation_list
    def get_action_id(self, atomic_action_id, object_id):
        """
        find the action id of an atomic action-object pair, returns None if not in the set
        :param action_id: int id of atomic action
        :param object_id: int id of object
        :return: int compound action id | None if not in set
        """
        idx = None
        for i, ao_pair in enumerate(self.ao_list):
            if ao_pair[0] == atomic_action_id and ao_pair[1] == object_id:
                idx = i + 1  # +1 to allow the NA first action label
                break
        return idx

    def get_video_annotations_table(self, video_idx):
        """
        fetch the annotation table of a specific video
        :param :  video_idx: index of the desired video
        :return: video annotations table
        """
        return self.cursor_annotations.execute('''SELECT * FROM annotations WHERE video_id = ?''', (video_idx,))

    def get_video_id_from_video_path(self, video_path, device='dev3'):
        """
        return video id for a given video path
        :param video_path: path to video (including device and image dir)
        :return: video_id: id of the video
        """
        # print(video_path)
        rows = self.cursor_vid.execute('''SELECT * FROM videos WHERE video_path = ? AND camera = ?''',
                                (video_path, device)).fetchall()
        # print(rows)
        if len(rows) > 1:
            raise ValueError("more than one video with the desired specs - check database")
        else:
            output = rows[0]["id"]
        return output

    def get_video_table(self, video_idx):
        """
        fetch the video information row from the video table in the database
        :param :  video_idx: index of the desired video
        :return: video information table row from the databse
        """
        return self.cursor_annotations.execute('''SELECT * FROM videos WHERE id = ?''', (video_idx,))
    
    


    def change_video(self, video_path):
        self.scan_path = video_path
    def display(self):
        """Load RGB data"""
        # True label
        data = np.load(self.logdir + '/true.npz')
        true = [data[key] for key in data.keys()][self.video_index]
        # Prediction
        data = np.load(self.logdir + '/prediction.npz')
        pred = [data[key] for key in data.keys()][self.video_index]
        # Intensity
        data = np.load(self.logdir + '/intensity.npz')
        intensity = [data[key] for key in data.keys()][self.video_index]
        min_intensity = np.amin(intensity)
        max_intensity = np.amax(intensity)
        # Normalize intensity to 0 - 1
        intensity = (intensity - min_intensity) / (max_intensity - min_intensity) * 5

        if len(pred) != len(intensity):
            raise IntensityLengthError("intensity length is not equal to prediction length...")
        video_path = os.path.join(self.scan_path, self.modality, 'images', 'scan_video.avi')
        cap = cv2.VideoCapture(video_path)
        self.get_label()
        # Check if camera opened successfully
        if (cap.isOpened()== False): 
            print("Error opening video stream or file")
        
        # Read until video is completed
        frame_ind = 0
        while(cap.isOpened()):
        # Capture frame-by-frame
            ret, img = cap.read()
            if not self.context:
                img = np.zeros(img.shape,dtype=np.uint8)
                img.fill(255)
            if ret == True:
                self.file_idx = frame_ind
                curr_inten = intensity[frame_ind]
                print(curr_inten)
                # Show object segmentation
                color_cat = {1: (255, 0, 0), 2: (0, 0, 255), 3: (0, 255, 0), 4: (127, 0, 127), 5: (127, 64, 0), 6: (64, 0, 127),
                     7: (64, 0, 64)}
                cat_dict = {1: 'table_top', 2: 'leg', 3: 'shelf', 4: 'side_panel', 5: 'front_panel', 6: 'bottom_panel',
                     7: 'rear_panel'}
                obj_data = self.seg()   # keys:"category_id", "image_id", "area", "bbox", "iscrowd", "frame_index"
                for i, obj in enumerate(obj_data):
                    if i < 6:
                        inten = curr_inten[i + 18]
                    else:
                        inten = 1
                    obj = obj_data[i]
                    cat_id = obj['category_id']
                    bbox = obj['bbox'] # <bb_left>, <bb_top>, <bb_width>, <bb_height>
                    x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
                    center_x = int(x + w / 2)
                    center_y = int(y + h / 2)
                    pt1 = (int(x), int(y))
                    pt2 = (int(x + w), int(y + h))
                    cv2.rectangle(img, pt1, pt2, color_cat[cat_id], thickness=2)
                    cv2.circle(img, (center_x, center_y), math.ceil((inten**2)), (255, 0, 0), thickness=-1, lineType=cv2.FILLED)
                # Show skeleton data on img
                j2d, _ = self.pose()
                skeleton_pairs = [(4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12, 11), (10, 9), (9, 8),
                        (11, 5), (8, 2), (5, 1), (2, 1), (0, 1), (15, 0), (14, 0), (17, 15),
                        (16, 14)]
                # print(skeleton_pairs)
                bbox = self.seg()
                bad_points_idx = [] # Points that are failed detected
                # print(curr_inten)
                # print("Skeleton pairs", skeleton_pairs)
                for i, (point, inten) in enumerate(zip(j2d, curr_inten)):
                    if not point[0] == 0 and not point[1] == 0:
                        if i == 4 or i == 7:
            
                            cv2.circle(img, (int(point[0]), int(point[1])), math.ceil(inten**2), (0, 165, 255), thickness=-1, lineType=cv2.FILLED)
                        else:
                            cv2.circle(img, (int(point[0]), int(point[1])), math.ceil(inten**2), (0, 255, 0), thickness=-1, lineType=cv2.FILLED)
                    else:
                        bad_points_idx.append(i)

                part_colors = utils.get_pose_colors(mode='bgr')

                for i, pair in enumerate(skeleton_pairs):
                    partA = pair[0]
                    partB = pair[1]
                    # if partA == 18 or partB == 18:
                    #     continue
                    if partA not in bad_points_idx and partB not in bad_points_idx:
                        # if j2d[partA] and j2d[partB]:
                        line_color = part_colors[i]

                        img = cv2.line(img, (int(j2d[partA][0]), int(j2d[partA][1])), 
                                            (int(j2d[partB][0]), int(j2d[partB][1])), line_color, 2)
                # Display the resulting frame

                # Define the text to be written
                txt = 'Action: ' + self.action_list[self.label[frame_ind]]
                true_txt = 'True: ' + self.action_list[true[frame_ind]]
                pred_txt = 'Pred: ' + self.action_list[pred[frame_ind]]
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
                cv2.putText(img, txt, position, font, fontScale, (255,0,0), thickness)
                cv2.putText(img, true_txt, (position[0], position[1] + 50), font, fontScale, (0, 255, 0), thickness)
                cv2.putText(img, pred_txt, (position[0], position[1] + 100), font, fontScale, color, thickness)
                print("img size:", img.shape)
                cv2.imshow('Frame',img)
            
                # Press Q on keyboard to  exit
                if cv2.waitKey(100) == ord('q'):
                    break
                frame_ind += 1
            # Break the loop
            else: 
                break
        
        # When everything done, release the video capture object
        cv2.destroyWindow('Frame')
        cap.release()

    def display_rgb(self):
        video_path = os.path.join(self.scan_path, self.modality, 'images', 'scan_video.avi')
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

    def depth(self):
        """Load depth data"""
        filename = os.path.join(self.scan_path, self.modality, 'depth', str(self.file_idx).zfill(6) + '.png')
        depth = cv2.imread(filename, cv2.IMREAD_ANYDEPTH).astype(np.float32)
    


    def pose(self):
        """Load pose data and append it to img"""
        

        # Read json file for each frame
        pose_json_filename = os.path.join(self.scan_path, self.modality, 'predictions/pose2d/openpose', 
                                            'scan_video_' + str(self.file_idx).zfill(12) + '_keypoints' + '.json')
        # print(pose_json_filename)
        # data = utils.read_pose_json(pose_json_filename)
        with open(pose_json_filename) as json_file:
            data = json.load(json_file)
        # print("pose cost:", time.time()-start)
        data = data['people']
        # If more than one person is detected, get active one
        if len(data) > 1:
            pose = self.get_active_person(data, center=(960, 540), min_bbox_area=20000)
        else:
            pose = np.array(data[0]['pose_keypoints_2d'])  # x,y,confidence
        
        pose = pose.reshape(-1, 3)  # x,y,confidence
        # print('pose.shape:', pose.shape)

        pose = np.delete(pose, 8, 0)

        j2d = np.array(pose)
        
        skeleton_pairs = utils.get_staf_skeleton()

        # Plot the joints

        return j2d, skeleton_pairs

        

    def seg(self):
        segment_path = os.path.join(self.scan_path, 'dev3', 'seg', str(self.file_idx) + '.json')


        if not os.path.exists(segment_path):
            segments = []
        else:
            segments = json.load(open(segment_path))
        return segments
        
        

       

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
    dataset_dir = '/media/zhihao/Chi_SamSungT7/IKEA_ASM'
    db_file = os.path.join(dataset_dir, 'ikea_annotation_db_full')
    env_lists = ['Kallax_Shelf_Drawer', 'Lack_Coffee_Table', 'Lack_Side_Table', 'Lack_TV_Bench']
    dev = 'dev3'
    test_filename = os.path.join(dataset_dir,'indexing_files', 'test_cross_env.txt')
    testset_video_list = get_list_from_file(test_filename)
    # print(testset_video_list)
    scan_name = None
    video_index = 60
    env_dir = os.path.join(dataset_dir, env_lists[0])
    item_list = os.listdir(env_dir)
    logdir = '../pose_based/log/EGCN_50_w_obj_A'
    # This is a extreme case
    # scan_name = '/media/zhihao/Chi_SamSungT7/IKEA_ASM/Kallax_Shelf_Drawer/0040_black_floor_09_04_2019_08_28_13_21'
    # scan_name = '/media/zhihao/Chi_SamSungT7/IKEA_ASM/Kallax_Shelf_Drawer/0043_oak_table_10_04_2019_08_28_15_29'
    scan_name = os.path.join(dataset_dir, testset_video_list[video_index])
    switcher = FunctionSwitcher(scan_path=scan_name, output_path=None, modality=dev, 
                            resize_factor=None, context=True, video_index=video_index, logdir=logdir)

    switcher.display()
    # switcher.display_rgb()
    # switcher.display_depth()


    # pesudo_image = switcher.get_pesudo_image()
    # pesudo_image = np.array(pesudo_image)
    # pesudo_image = np.transpose(pesudo_image, (1, 0, 2))

    # cv2.imshow('pesudo_image', pesudo_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
        



