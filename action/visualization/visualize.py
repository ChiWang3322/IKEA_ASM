import os
import shutil
import cv2
import vis_utils
import numpy as np
import json
import random
import sys
sys.path.append('../')
import utils

#TODO implement adjust depth to rgb image size

class FunctionSwitcher:

    def __init__(self, scan_path, output_path=None, modality='dev3', resize_factor=None):
        self.scan_path = scan_path
        self.output_path = output_path
        # os.makedirs(output_path, exist_ok=True)
        self.modality = modality
        # self.name_mapping_dict = name_mapping_dict
        self.resize_factor = resize_factor
        self.file_idx = 0

    def change_video(self, video_path):
        self.scan_path = video_path
    def display(self):
        """Load RGB data"""

        video_path = os.path.join(self.scan_path, self.modality, 'images', 'scan_video.avi')
        cap = cv2.VideoCapture(video_path)
        
        # Check if camera opened successfully
        if (cap.isOpened()== False): 
            print("Error opening video stream or file")
        
        # Read until video is completed
        frame_ind = 0
        while(cap.isOpened()):
        # Capture frame-by-frame
            ret, img = cap.read()
            if ret == True:
                self.file_idx = frame_ind
                
                # Show object segmentation
                color_cat = {1: (255, 0, 0), 2: (0, 0, 255), 3: (0, 255, 0), 4: (127, 0, 127), 5: (127, 64, 0), 6: (64, 0, 127),
                     7: (64, 0, 64)}
                cat_dict = {1: 'table_top', 2: 'leg', 3: 'shelf', 4: 'side_panel', 5: 'front_panel', 6: 'bottom_panel',
                     7: 'rear_panel'}
                obj_data = self.seg()   # keys:"category_id", "image_id", "area", "bbox", "iscrowd", "frame_index"
                for obj in obj_data:
                    cat_id = obj['category_id']
                    bbox = obj['bbox'] # <bb_left>, <bb_top>, <bb_width>, <bb_height>
                    x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
                    pt1 = (int(x), int(y))
                    pt2 = (int(x + w), int(y + h))
                    cv2.rectangle(img, pt1, pt2, color_cat[cat_id], thickness=2)

                # Show skeleton data on img
                j2d, _ = self.pose()
                skeleton_pairs = [(4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12, 11), (10, 9), (9, 8),
                        (11, 5), (8, 2), (5, 1), (2, 1), (0, 1), (15, 0), (14, 0), (17, 15),
                        (16, 14)]
                print(skeleton_pairs)
                bbox = self.seg()
                bad_points_idx = [] # Points that are failed detected
                # print("Skeleton pairs", skeleton_pairs)
                for i, point in enumerate(j2d):
                    if not point[0] == 0 and not point[1] == 0:
                        cv2.circle(img, (int(point[0]), int(point[1])), 4, (255, 0, 0), thickness=-1, lineType=cv2.FILLED)
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
                                            (int(j2d[partB][0]), int(j2d[partB][1])), line_color, 3)
                # Display the resulting frame
                
                cv2.imshow('Frame',img)
            
                # Press Q on keyboard to  exit
                if cv2.waitKey(40) == ord('q'):
                    break
                frame_ind += 1
            # Break the loop
            else: 
                break
        
        # When everything done, release the video capture object
        cv2.destroyWindow('Frame')
        cap.release()



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
            pass
        else:
            segments = json.load(open(segment_path))
        return segments
        
        

        # all_segments = json.load(open(segment_path))
        # fid_track = open(tracking_path)
        # tracking_results = str.split(fid_track.read(), '\n')

        # track_id = []
        # dict_tracks = {}
        # for track in tracking_results:
        #     if track != "":
        #         track_id.append(int(str.split(track, ' ')[-3]))
        #         items = str.split(track, ' ')
        #         if items[-2] not in dict_tracks:
        #             dict_tracks[items[-2]] = []
        #         dict_tracks[items[-2]].append([items[0:5], items[-1]])

        # # Obtain Unique colors for each part
        # dict_colors = {}
        # max_part = np.max(np.unique(track_id))
        # r = random.sample(range(0, 255), max_part)
        # g = random.sample(range(0, 255), max_part)
        # b = random.sample(range(0, 255), max_part)
        # for part_id in np.unique(track_id):
        #     dict_colors[str(part_id)] = (int(r[part_id - 1]), int(g[part_id - 1]), int(b[part_id - 1]))

        # all_segments_dict = {}

        # for item in all_segments['annotations']:
        #     if item['image_id'] not in all_segments_dict:
        #         all_segments_dict[item['image_id']] = []
        #     all_segments_dict[item['image_id']].append(item)

        # fname = str(self.file_idx).zfill(6) + '.jpg'
        # image_id = self.find_id(fname, all_segments)
        # fname_id = int(str.split(fname, '.')[0])

        # predictions = vis_utils.overlay_segmentation_mask(img, all_segments_dict[image_id], dict_tracks[str(fname_id)],
        #                                              dict_colors, color_cat, cat_dict)

        # if not self.resize_factor is None:
        #     h, w, c = predictions.shape
        #     h, w = (int(h / self.resize_factor), int(w / self.resize_factor))
        #     predictions = cv2.resize(predictions, dsize=(w, h))  # resizing the images

        # cv2.imwrite(output_filename, predictions)
        # print(" Saved object segmentation to {}".format( output_filename))

    def find_id(self, image_name, test_data):
        pass
        # for item in test_data['images']:
        #     if item['file_name'].find(image_name) != -1:
        #         return item['id']
        # return -1
    
    def get_active_person(self, data, center=(960, 540)):
        """
        Select the active skeleton in the scene by applying a heuristic of findng the one closest to the center of the frame
        Parameters
        ----------
        data : pose data extracted from json file

        Returns
        -------
        pose: skeleton of the active person in the scene
        """
        pose = None
        min_dtc = 0 # dtc = distance to center
        for person in data['people']:
            current_pose = person['pose_keypoints_2d']
            dtc = self.compute_skeleton_distance_to_center(current_pose, center=center)
            if dtc < min_dtc:
                pose = current_pose
                min_dtc = dtc
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
        diff = skeleton - np.tile(center, len(skeleton[idx]))
        distances = np.sqrt(np.mean(diff ** 2))
        mean_distance = np.mean(distances)

        return mean_distance


if __name__ == '__main__':
    dataset_dir = '/media/zhihao/Chi_SamSungT7/IKEA_ASM'
    env_lists = ['Kallax_Shelf_Drawer', 'Lack_Coffee_Table', 'Lack_Side_Table', 'Lack_TV_Bench']
    dev = 'dev3'
    scan_name = None
    env_dir = os.path.join(dataset_dir, env_lists[0])
    item_list = os.listdir(env_dir)
    scan_name = os.path.join(env_dir, item_list[0])
    switcher = FunctionSwitcher(scan_path=scan_name, output_path=None, modality=dev, resize_factor=None)
    switcher.display()
    # for env in env_lists:
    #     env_dir = os.path.join(dataset_dir, env)
    #     item_list = os.listdir(env_dir)
    #     for item in item_list:
    #         scan_name = os.path.join(env_dir, item)
    #         print("Scan_name:", scan_name)
    #         switcher.change_video(scan_name)
    #         switcher.display()
    #         input("Press Enter to Continue...")
