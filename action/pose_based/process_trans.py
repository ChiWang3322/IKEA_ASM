import json
import glob
import os
import re
import copy
import cv2
import numpy
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import csv
import cv2
import open3d as o3d
from tqdm import tqdm
import shutil

def numericalSort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def parse_dir(root_dir):
    parse_result = []
    for file in sorted(glob.glob(os.path.join(root_dir, "*.json")), key=numericalSort):
        file_path = file
        parse_result.append(file_path)
        # print(file)
    return parse_result

def parse_image(root_dir):
    parse_result = []
    for file in sorted(glob.glob(os.path.join(root_dir, "*.png")), key=numericalSort):
        file_path = file
        parse_result.append(file_path)
        # print(file)
    return parse_result

def caculate_center(object_3d_bounding_box):
    x = (object_3d_bounding_box["x0"] + object_3d_bounding_box["x1"])/2
    y = (object_3d_bounding_box["y0"] + object_3d_bounding_box["y1"])/2
    z = (object_3d_bounding_box["z0"] + object_3d_bounding_box["z1"])/2
    center = (x, y, z)
    return center


def preprocess_point_cloud(pcd, voxel_size):
    # print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    # print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    # print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


def get_3d_object_center(data_2d_tmp, depth_image, rgb_image):
    points_list = []
    fx = 540.994
    fy = 540.994
    cx = 320.000
    cy = 240.000

    height, width, _ = depth_image.shape
    
    for object_idx, object_2d in enumerate(data_2d_tmp):
        # object_2d:<bb_left>, <bb_top>, <bb_width>, <bb_height>
        box_width = int(object_2d[2])
        box_high = int(object_2d[3])
        center_x = object_2d[0] + box_width / 2
        center_y = object_2d[1] * box_high / 2


        # center_x = object_2d['bounding_box']['x'] * width
        # center_y = object_2d['bounding_box']['y'] * height
        # box_width = object_2d['bounding_box']['w'] * width
        # box_high = object_2d['bounding_box']['h'] * height

        points = o3d.geometry.PointCloud()

        vi = int(center_x)
        ui = int(center_y)

        if depth_image[vi, ui] > 0:
            # Z = depth_image[vi, ui] / 1000  # millimeter
            Z = depth_image[vi, ui]  # millimeter

            u, v, Z = vi, ui, Z  # ui, height - vi, -Z

            X = (u - cx) * Z / fx
            Y = (v - cy) * Z / fy
            # points_before_clustering.append(np.array([X, Y, Z]))

            # points.points.append(np.array([X, Y, Z]))  # add point cloud to the list
            # points.colors.append(rgb_image[vi, ui] / 255.0)  # rgb_image[vi, ui] / 255.0)

            points.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

            points_list.append(np.array([X, Y, Z]))
    return points_list


def get_3d_object(data_2d_tmp, depth_image, rgb_image):
    points_list = []
    fx = 540.994
    fy = 540.994
    cx = 320.000
    cy = 240.000
    height, width= depth_image.shape
    # print('test.....',depth_image.shape)
    # print("Len data_2d_tmp:", len(data_2d_tmp))
    for object_idx, object_2d in enumerate(data_2d_tmp):
        # <bb_left>, <bb_top>, <bb_width>, <bb_height>
        object_2d = object_2d['bbox']
        box_width = object_2d[2] 
        box_high = object_2d[3] 
        center_x = object_2d[0] + box_width / 2
        center_y = object_2d[1] + box_high / 2


        points = o3d.geometry.PointCloud()

        start_point = int(object_2d[0]), int(object_2d[1])
        end_point = int(object_2d[0] + box_width), int(object_2d[1] + box_high)
        if end_point[0] > width:
            end_point[0] = width
        if end_point[1] > height:
            end_point[1] = height
        # print("End point:", end_point)
        for vi in range(start_point[1], end_point[1]):
            for ui in range(start_point[0], end_point[0]):
                if depth_image[vi, ui] > 0:
                    # Z = depth_image[vi, ui] / 1000  # millimeter
                    Z = depth_image[vi, ui]  # millimeter

                    u, v, Z = ui, vi, Z  # ui, height - vi, -Z

                    X = (u - cx) * Z / fx
                    Y = (v - cy) * Z / fy
                    # points_before_clustering.append(np.array([X, Y, Z]))

                    points.points.append(np.array([X, Y, Z]))  # add point cloud to the list
                    points.colors.append(rgb_image[vi, ui] / 255.0)  # rgb_image[vi, ui] / 255.0)

        points.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

        # print("Statistical oulier removal")
        cl, ind = points.remove_statistical_outlier(nb_neighbors=50, std_ratio=1.0)
        inlier_cloud = points.select_by_index(ind)
        points_list.append(inlier_cloud)
    return points_list


def calculate_trans(curr_data_2d_tmp, past_data_2d_tmp, curr_depth_image, past_depth_image, curr_rgb_image, past_rgb_image):
    # This is a function to calculate transformation matrix

    # Inputs:
    #     curr_data_2d_tmp(list): object data of current frame
    #     past_data_2d_tmp(list): object data of past frame
    #
    # Outputs:
    #     list[]:transformation matrix


    lens = min(len(curr_data_2d_tmp), len(past_data_2d_tmp))
    target_list = get_3d_object(curr_data_2d_tmp, curr_depth_image, curr_rgb_image)
    source_list = get_3d_object(past_data_2d_tmp, past_depth_image, past_rgb_image)
    transformation_matrix = []
    for j in range(0, lens):
        source = source_list[j]
        target = target_list[j]
        voxel_size = 0.005  # means 5cm for this dataset
        source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
        target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)

        threshold = 0.004
        icp_iteration = 50  # default 100
        save_image = False
        # save_image = True

        res = np.identity(4)
        loss = o3d.pipelines.registration.TukeyLoss(k=0.1)
        for i in range(icp_iteration):
            reg_p2l = o3d.pipelines.registration.registration_icp(
                source_down, target_down, threshold, np.identity(4),
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1))
            reg_p2p = o3d.pipelines.registration.registration_icp(
                source_down, target_down, threshold, reg_p2l.transformation,
                o3d.pipelines.registration.TransformationEstimationPointToPoint())
            # reg_ransac = execute_global_registration(source_down, target_down, source_fpfh,
            #                             target_fpfh, voxel_size)
            # reg_fast = execute_fast_global_registration(source_down, target_down, source_fpfh,
            #                                  target_fpfh, voxel_size)
            # R, T = registration_RANSAC(source_down, target_down, source_fpfh,
            #                            target_fpfh)
            # print("curr R: curr T: ", '\n', R, '\n', T)
            source_down.transform(reg_p2p.transformation)
            # source_down.transform(reg_ransac.transformation)
            # source_down.transform(reg_fast.transformation)
            # source_down.transform(current_transformation)
            source.transform(reg_p2p.transformation)
            # vis.update_geometry(source_down)
            # vis.poll_events()
            # vis.update_renderer()
            # print(reg_p2p.transformation)

            res = res.dot(reg_p2p.transformation)

        # Colored Point Cloud Registration Revisited, ICCV 2017
        voxel_radius = [0.04, 0.02, 0.01, 0.001]
        max_iter = [50, 30, 20, 14]
        # current_transformation = np.identity(4)
        current_transformation = reg_p2p.transformation
        # print("3. Colored point cloud registration")
        for scale in range(4):
            iter = max_iter[scale]
            radius = voxel_radius[scale]
            # print([iter, radius, scale])

            # print("3-1. Downsample with a voxel size %.2f" % radius)
            source_down = source.voxel_down_sample(radius)
            target_down = target.voxel_down_sample(radius)

            # print("3-2. Estimate normal.")
            source_down.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
            target_down.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))

            # print("3-3. Applying colored point cloud registration")
            result_icp = o3d.pipelines.registration.registration_icp(
                source_down, target_down, radius, current_transformation,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=iter))
            # result_icp = o3d.pipelines.registration.registration_colored_icp(
            #     source_down, target_down, radius, current_transformation,
            #     o3d.pipelines.registration.TransformationEstimationForColoredICP(),
            #     o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
            #                                                       relative_rmse=1e-6,
            #                                                       max_iteration=iter))
            current_transformation = result_icp.transformation
        source_down.transform(reg_p2p.transformation)
        res = current_transformation
        transformation_matrix.append(res)

    return transformation_matrix


def transformation(data_tmp, curr_trans, i_frame, righthand_gt_label = 0, lefthand_gt_label = 0, skeleton_3d_position = 0):
    frame_dic_list = []
    lens = len(curr_trans)
    for object_idx, object in enumerate(data_tmp):
        object_instance = object['instance_name']
        object_class_index = object['class_index']
        current_center = caculate_center(object['bounding_box'])

        # vis



        past_center = caculate_center(object['past_bounding_box'])
        tx = current_center[0] - past_center[0]
        ty = current_center[1] - past_center[1]
        tz = current_center[2] - past_center[2]
        transformation_matrix = [[1, 0, 0, tx], [0, 1, 0, ty], [0, 0, 1, tz], [0, 0, 0, 1]]
        if lens > object_idx:
            transformation_matrix = curr_trans[object_idx].tolist()
        time = 1/30 * i_frame     # 30 fps
        dict_tmp = {"object_instance": object_instance,
                    "object_class_index": object_class_index,
                    "current_center": current_center,
                    "past_center": past_center,
                    "transformation_matrix": transformation_matrix,
                    "time": time}
                    # "righthand_gt_label": righthand_gt_label,
                    # "lefthand_gt_label": lefthand_gt_label
                    # "skeleton_3d_position": skeleton_3d_position
                    #
        frame_dic_list.append(dict_tmp)
    return frame_dic_list

def extend_labels(labels_list):
    if None in labels_list:
        l = [i for i, v in enumerate(labels_list) if v == None][0]
        labels_list[l] = 0

    frame_length = int(max(labels_list) + 1)
    # left_hand_ground_truth_labels = np.zeros((1, frame_length))
    hand_ground_truth_labels = np.zeros((1, frame_length))

    for i, frame_idx in enumerate(labels_list):
        if i == 2:
            hand_ground_truth_labels[0][labels_list[0]:(labels_list[2] + 1)] = labels_list[1]
        elif i % 2 == 0:
            hand_ground_truth_labels[0][(labels_list[i - 2] + 1):(labels_list[i] + 1)] = \
            labels_list[i - 1]

    return hand_ground_truth_labels[0]


def depth_images_transformation(depth_images_parse_result):
    depth_images = []
    for i, depth_image in enumerate(depth_images_parse_result):
        depth_img = cv2.imread(depth_image)
        extracted_depth_img = depth_img[:, :, 2] + 255.0 * depth_img[:, :, 1]
        depth_images.append(extracted_depth_img)
    return depth_images


def rgb_images_transformation(rgb_images_parse_result):
    rgb_images = []
    for i, rgb_image in enumerate(rgb_images_parse_result):
        rgb_img = cv2.imread(rgb_image)
        rgb_images.append(rgb_img)
    return rgb_images


def map2d_skeleton_to3D(skeleton_data_2d, i, depth_images):
    "camera parameters"
    fx = 540.994
    fy = 540.994
    cx = 320.000
    cy = 240.000

    "Generate 2d skeleton position dict"
    skeletion_3d_list = []
    for j, position in enumerate(skeleton_data_2d[0]):
        joint_class = skeleton_data_2d[0][position]["label"]
        pixel_x = int(skeleton_data_2d[0][position]["x"])
        pixel_y = int(skeleton_data_2d[0][position]["y"])

        joint_depth = depth_images[i][pixel_y, pixel_x]

        "Map joint from 2d to 3d"
        joint_3d_Z = joint_depth
        u, v, Z = pixel_x, pixel_y, joint_3d_Z
        joint_3d_x = (u - cx) * Z / fx
        joint_3d_y = (v - cy) * Z / fy

        "Map 2d skeleton position dict to 3d"
        skeletion_3d_tmp_dict = {"joint_class": joint_class,
                            "pixel_x": joint_3d_x,
                            "pixel_y": joint_3d_y,
                            "joint_depth": joint_3d_Z}
        skeletion_3d_list.append(skeletion_3d_tmp_dict)

    return skeletion_3d_list



def get_trans(scan_name, step = 1):
    depth_video = os.path.join(scan_name, 'dev3', 'depth', 'scan_video.avi')
    rgb_video = os.path.join(scan_name, 'dev3', 'images', 'scan_video.avi')
    error_text_path = '/media/zhihao/Chi_SamSungT7/IKEA_ASM/trans_error_file.txt'
    error_flag = False
    print(rgb_video)
    print(depth_video)
    depth = cv2.VideoCapture(depth_video)
    rgb = cv2.VideoCapture(rgb_video)
    trans_m = []
    frame_count = 0
    if os.path.exists(os.path.join(scan_name, 'dev3', 'trans_' + str(step))):
        shutil.rmtree(os.path.join(scan_name, 'dev3', 'trans_' + str(step)))
    os.makedirs(os.path.join(scan_name, 'dev3', 'trans_' + str(step)), exist_ok=True)
    while depth.isOpened() and rgb.isOpened():
        # print("Current frame:", frame_count)
        ret1, frame1 = depth.read()
        ret2, frame2 = rgb.read()
        trans_path = os.path.join(scan_name, 'dev3', 'trans_' + str(step), str(frame_count)+'.json')
        # print('Current frame:', frame_count)
        if not ret1 or not ret2:
            print(ret1)
            print(ret2)
            print('Not ret1 or not ret2, break...')
            break

        # Check if this folder has been processed
        # num_frames = int(rgb.get(cv2.CAP_PROP_FRAME_COUNT))
        # trans_list = os.listdir(os.path.join(scan_name, 'dev3', 'trans_' + str(step)))
        # if len(trans_list) == num_frames:
        #     break
        # Check if this frame has been processed

        # if os.path.exists(trans_path):
        #     frame_count += 1
        #     continue
        curr_depth_image = cv2.resize(frame1, (frame2.shape[1], frame2.shape[0]))
        # Load as gray image
        tmp = curr_depth_image[:, :, 2] + 255.0 * curr_depth_image[:, :, 1]
        curr_depth_image = tmp
        curr_rgb_image = frame2
        obj_path = os.path.join(scan_name, 'dev3', 'seg', str(frame_count)+'.json')
        curr_data_2d_tmp = json.load(open(obj_path))

        if frame_count % step != 0:
            frame_count += 1
            continue

        # Store the past data
        


        if not frame_count == 0:
            
            # print("curr_depth:", curr_depth_image.shape)
            # print("curr_rgb:", curr_rgb_image.shape)
            print("Processing registration between frame {} and frame {}".format(frame_count, frame_count - step))
            try:
                trans_m = calculate_trans(curr_data_2d_tmp, past_data_2d_tmp, curr_depth_image, 
                                        past_depth_image, curr_rgb_image, past_rgb_image)

            except:
                print("Encounter error at frame{}, but continue...".format(frame_count))
                error_flag = True
                break

        frame_count += 1
        past_depth_image = curr_depth_image.copy()
        past_rgb_image = curr_rgb_image.copy()
        past_data_2d_tmp = curr_data_2d_tmp


        print("Number of objects:", len(curr_data_2d_tmp))
        print("shape of trans_m:", np.shape(trans_m))
        print("Trans M:", trans_m)
        # Convert numpy array to list trans_m:[ndarray, ndarray...]
        for i in range(len(trans_m)):
            trans_m[i] = trans_m[i].tolist()

        with open(trans_path, 'w') as f:
            json.dump(trans_m, f)
        




        # cv2.imshow('Depth and RGB', frame_combined)
        
        # if cv2.waitKey(20) & 0xFF == ord('q'):
        #     break

    depth.release()
    rgb.release()
    if error_flag:
        with open(error_text_path, 'a') as f:
            # Write the text to the file
            f.write(scan_name + '\n')
        
    # print("{} transformation matrix extracted successfully...".format(scan_name))

        # font = cv2.FONT_HERSHEY_SIMPLEX
        
        # org
        # org = (50, 50)
        
        # fontScale
        # fontScale = 1
        
        # # Blue color in BGR
        # color = (255, 0, 0)
        
        # Line thickness of 2 px
        # thickness = 2

        # rgb_size = frame2.shape

        # frame1_resized = cv2.resize(frame1, (int(rgb_size[1]/2), int(rgb_size[0]/2)))
        # frame2_resized = cv2.resize(frame2, (int(rgb_size[1]/2), int(rgb_size[0]/2)))
        
        # frame_combined = cv2.vconcat([frame1_resized, frame2_resized])

        # a_str = str(trans_m)
        # a_str_with_brackets = '[' + a_str[1:-1] + ']'
        # cv2.putText(frame_combined, a_str_with_brackets, org, font, fontScale, color, thickness, cv2.LINE_AA)


def matrix_to_quaternion(matrix):

    R = matrix[:3, :3]
    trace = np.trace(R)
    
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
    
    return np.array([w, x, y, z])






if __name__ == '__main__':
    # read rgb video and depth video
    dataset_dir = '/media/zhihao/Chi_SamSungT7/IKEA_ASM'
    env_lists = ['Kallax_Shelf_Drawer', 'Lack_Coffee_Table', 'Lack_Side_Table', 'Lack_TV_Bench']
    dev = 'dev3'
    # env_dir = os.path.join(dataset_dir, env_lists[0])
    # scan_name = '/media/zhihao/Chi_SamSungT7/IKEA_ASM/Lack_Side_Table/0039_white_floor_08_04_2019_08_28_10_40'
    step = 100
    for env in env_lists:
        env_dir = os.path.join(dataset_dir, env)
        item_list = os.listdir(env_dir)
        for item in tqdm(item_list):
            scan_name = os.path.join(dataset_dir, env, item)
            print("Processing dir:", scan_name)
            # print(scan_name)
            get_trans(scan_name, step = step)




    


