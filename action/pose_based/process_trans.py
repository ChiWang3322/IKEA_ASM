import json
import glob
import os
import re
import copy

import numpy
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import csv
import cv2
import open3d as o3d

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
    height, width = depth_image.shape
    for object_idx, object_2d in enumerate(data_2d_tmp):

        center_x = object_2d['bounding_box']['x'] * width
        center_y = object_2d['bounding_box']['y'] * height
        box_width = object_2d['bounding_box']['w'] * width
        box_high = object_2d['bounding_box']['h'] * height

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
    height, width = depth_image.shape
    for object_idx, object_2d in enumerate(data_2d_tmp):

        center_x = object_2d['bounding_box']['x'] * width
        center_y = object_2d['bounding_box']['y'] * height
        box_width = object_2d['bounding_box']['w'] * width
        box_high = object_2d['bounding_box']['h'] * height

        points = o3d.geometry.PointCloud()

        start_point = int(center_x - box_width / 2), int(center_y - box_high / 2)
        end_point = int(center_x + box_width / 2), int(center_y + box_high / 2)

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


def skeleton_visulisation(skeleton_3d_position_list):
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection='3d')

    X = [1,2,3,4,5,6,7,8]
    Y = [5, 7, 8, 9, 10, 11, 7, 8]
    Z = [3,7,3,11,9,2,10,11]

    # for i, frame_sk in enumerate(skeleton_3d_position_list):


    ax.plot(X, Y, Z)  # plot the point (2,3,4) on the figure

    plt.show()



def get_video_transformation_list(data_dir, subject_id):
    'Labels Ground Truth'
    # labels_file_path = "/home/yuankai/state_of_the_art/har/marsil/model-exploration/src/simulation/kit_bimanual/labels/take_0.json"
    labels_file_path = f"{data_dir}/labels/take_{subject_id}.json"
    labels_file = open(labels_file_path)
    labels = json.load(labels_file)
    labels_file.close()

    original_right_hand_ground_truth_labels = labels['right_hand']
    original_left_hand_ground_truth_labels = labels['left_hand']

    righthand_ground_truth_labels = extend_labels(original_right_hand_ground_truth_labels)
    lefthand_ground_truth_labels = extend_labels(original_left_hand_ground_truth_labels)

    """Labels Ground Truth"""
    """ activity label"""
    activity_gt_label = "cooking_with_bowls"

    """ right/left hand gt"""
    # labels_file_path = "/home/yuankai/state_of_the_art/har/marsil/model-exploration/src/simulation/kit_bimanual/labels/take_0.json"
    labels_file_path = "sample_data/labels/take_0.json"

    "Skeleton position 2d to 3d projection"
    # skeleton_file_path = "/home/yuankai/state_of_the_art/har/marsil/model-exploration/src/simulation/kit_bimanual/body_pose"
    skeleton_file_path = f"{data_dir}/body_pose"
    skeleton_parse_result = parse_dir(skeleton_file_path)

    "Extract depth info"
    # depth_image_path = "/home/yuankai/datasets/kit_bimanual/bimacs_rgbd_data/subject_1/task_2_k_cooking_with_bowls/take_0/depth"
    depth_image_path = f"{data_dir}/depth"
    depth_images_list = []
    depth_images_parse_result = []
    for i in range(5):
        depth_image_path_tmp = depth_image_path + "/chunk_" + str(i)
        depth_images_parse_result += parse_image(depth_image_path_tmp)
    depth_images = depth_images_transformation(depth_images_parse_result)

    "'Map 2D skeleton to 3D'"
    skeleton_3d_position_list = []
    for i, file in enumerate(skeleton_parse_result):
        skeleton_data_2d = open(file)
        skeleton_tmp = json.load(skeleton_data_2d)
        skeleton_3d_position = map2d_skeleton_to3D(skeleton_tmp, i, depth_images)
        skeleton_3d_position_list.append(skeleton_3d_position)
        skeleton_data_2d.close()

    'Object positions'
    # objects_file_path = "/home/yuankai/state_of_the_art/har/marsil/model-exploration/src/simulation/kit_bimanual/3d_objects"
    objects_file_path = f"{data_dir}/3d_objects"
    objects_2d_file_path = f"{data_dir}/2d_objects"
    object_spatial_relations_list = []
    video_transformation_list = []
    object_parse_result = parse_dir(objects_file_path)
    object_2d_parse_result = parse_dir(objects_2d_file_path)

    "Build the Train Datset"
    for i, file in enumerate(object_parse_result):
        file_data = open(file)
        data_tmp = json.load(file_data)
        righthand_gt_label = righthand_ground_truth_labels[i]
        lefthand_gt_label = lefthand_ground_truth_labels[i]
        dict_tmp = transformation(data_tmp, i, righthand_gt_label, lefthand_gt_label, skeleton_3d_position_list[i])
        video_transformation_list.append(dict_tmp)
        file_data.close()

    # print(video_transformation_list)
    return video_transformation_list


def read_file(bounding_box_filepath):
    with open(bounding_box_filepath, 'r') as f:
        data = json.load(f)
    # print(data)
    # print(data.len())
    # print(type(data))
    length = len(data)
    bounding_box_2d = []

    for num in range(0, length):
        sub_data = data[num]
        # print(type(sub_data))
        adds = []
        adds.append(sub_data['candidates'][0]['class_index'])
        adds.append(sub_data['bounding_box']['h'])
        adds.append(sub_data['bounding_box']['w'])
        adds.append(sub_data['bounding_box']['x'])
        adds.append(sub_data['bounding_box']['y'])
        bounding_box_2d.append(adds)

    return bounding_box_2d


def vis_cp(objs_dict_tmp, skeleton_list):

    FOR1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=20, origin=[0, 0, 0])

    out_point_pf = o3d.geometry.PointCloud()
    out_center_pf = o3d.geometry.PointCloud()

    # vis.add_geometry(out_point_pf)

    # add obj-center

    # from 2d
    out_obj_pf = []
    # center_list = method_2d23d(jsonPath, rgb_filepath, depth_filepath)
    obj_pc_list = get_3d_object(curr_data_2d_tmp, curr_depth_image, curr_rgb_image)
    obj_center_list = get_3d_object_center(curr_data_2d_tmp, curr_depth_image, curr_rgb_image)
    # print("obj_pc_list:", obj_pc_list)

    for idx, obj in enumerate(obj_center_list):
        out_center_pf.points.append(obj)

    for idx, obj in enumerate(obj_pc_list):
        out_obj_pf.append(obj)

    # print("out_center_pf:", np.array(out_center_pf.points))
    # from 3d
    # for idx, obj in enumerate(objs_dict_tmp):
    #     curr_center = obj['current_center']
    #     # "object_instance": "LeftHand_5"
    #     # RightHand_4
    #     if obj['object_instance'] == "LeftHand_5" or obj['object_instance'] == "RightHand_4":
    #         continue
    #     out_center_pf.points.append(np.array(curr_center))
    #     # vis.add_geometry(curr_center)
    #     # vis.update_geometry(out_point_pf)
    #     # vis.update_geometry(curr_center)
    #     # o3d.visualization.draw_geometries([out_point_pf])
    # # o3d.visualization.draw_geometries([FOR1, out_point_pf])
    # # o3d.visualization.draw_geometries([out_point_pf])
    # # out_center_pf.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    # add joint-posi
    delete_flag = []
    for idx, joint in enumerate(skeleton_list):
        X = joint['pixel_x']
        Y = joint['pixel_y']
        Z = joint['joint_depth']
        class_label = joint['joint_class']
        class_id = POSE_BODY_25_BODY_PARTS['{}'.format(class_label)]
        if Z == 0:
            delete_flag.append(idx)
            # last_pos = out_point_pf.points[-1]
        #     out_point_pf.points.append(np.array(last_pos))
        # else:
        out_point_pf.points.append(np.array([X, Y, Z]))
        # o3d.visualization.draw_geometries([out_point_pf])
        # vis.update_geometry(out_point_pf)
    # o3d.visualization.draw_geometries([FOR1, out_point_pf])
    out_point_pf.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    # print(numpy.asarray(out_point_pf.points))

    corresponds_1 = [14, 19, 18, 6, 16, 21, 12, 13, 5, 20, 7, 8, 1, 0, 11, 22, 17, 3, 15, 24, 9, 10, 2, 23, 4]
    corresponds_2 = [[1, 8], [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [8, 9], [9, 10], [10, 11], [8, 12], [12, 13], [13, 14], [1, 0],
                     [0, 15], [15, 17], [0, 16], [16, 18], [2, 17], [5, 18], [14, 19], [19, 20], [14, 21], [11, 22], [22, 23], [11, 24]]

    lines = [[12, 11], [12, 22], [12, 8], [22, 17], [17, 24], [8, 3], [3, 10], [11, 20], [20, 21], [21, 14], [11, 6],
             [6, 7], [7, 0], [12, 13], [13, 18], [18, 16], [13, 4], [4, 2], [22, 16], [8, 2], [0, 1], [1, 9], [0, 5],
             [14, 15], [15, 23], [14, 19]]
    # lines_p = copy.deepcopy(lines)
    for i, flag in enumerate(delete_flag):
        for j, pair in enumerate(lines):
            if pair[0] == flag or pair[1] == flag:
                # lines_p.remove(pair)
                lines.pop(j)
        # out_point_pf.points.pop(i)
        # idx = lines.index(flag)
        # del lines_p[idx]

    # for i, flag in enumerate(delete_flag):
    #     out_point_pf.points.pop(i)

    # colors = [[0, 0, 1] for i in range(len(lines))]  # Default blue

    # line between joints
    line_pcd = o3d.geometry.LineSet()
    # line_pcd = o3d.geometry.LineSet(
    #             points=o3d.utility.Vector3dVector(out_point_pf),
    #             lines=o3d.utility.Vector2iVector(lines))
    line_pcd.lines = o3d.utility.Vector2iVector(lines)
    # line_pcd.colors = o3d.Vector3dVector(colors)
    line_pcd.points = out_point_pf.points
    # del line_pcd.points[delete_flag]
    # print("out_point_pf:", np.array(out_point_pf.points))

    # for idx, point in enumerate(out_point_pf.points):


    # o3d.visualization.draw_geometries([out_point_pf, line_pcd])

    # vis.poll_events()
    # vis.update_renderer()
    return out_point_pf, line_pcd, out_center_pf, out_obj_pf


if __name__ == '__main__':
    DATA_DIR = 'sample_data'
    'Labels Ground Truth'
    # labels_file_path = "/home/yuankai/state_of_the_art/har/marsil/model-exploration/src/simulation/kit_bimanual/labels/take_0.json"
    labels_file_path = f"{DATA_DIR}/labels/take_0.json"

    labels_file = open(labels_file_path)
    labels = json.load(labels_file)
    labels_file.close()

    original_right_hand_ground_truth_labels = labels['right_hand']
    original_left_hand_ground_truth_labels = labels['left_hand']

    righthand_ground_truth_labels = extend_labels(original_right_hand_ground_truth_labels)
    lefthand_ground_truth_labels = extend_labels(original_left_hand_ground_truth_labels)


    "Skeleton position 2d to 3d projection"
    # skeleton_file_path = "/home/yuankai/state_of_the_art/har/marsil/model-exploration/src/simulation/kit_bimanual/body_pose"

    # skeleton_file_path = "sample_data/body_pose"

    skeleton_file_path = f"{DATA_DIR}/body_pose"

    skeleton_parse_result = parse_dir(skeleton_file_path)

    "Extract depth info"
    # depth_image_path = "/home/yuankai/datasets/kit_bimanual/bimacs_rgbd_data/subject_1/task_2_k_cooking_with_bowls/take_0/depth"

    # depth_image_path = "sample_data/depth"

    depth_image_path = f"{DATA_DIR}/depth"

    depth_images_list = []
    depth_images_parse_result = []
    for i in range(5):
        depth_image_path_tmp = depth_image_path + "/chunk_" + str(i)
        depth_images_parse_result += parse_image(depth_image_path_tmp)
    depth_images = depth_images_transformation(depth_images_parse_result)

    rgb_image_path = f"{DATA_DIR}/rgb"

    rgb_images_list = []
    rgb_images_parse_result = []
    for i in range(5):
        rgb_image_path_tmp = rgb_image_path + "/chunk_" + str(i)
        rgb_images_parse_result += parse_image(rgb_image_path_tmp)
    rgb_images = rgb_images_transformation(rgb_images_parse_result)

    "'Map 2D skeleton to 3D'"
    skeleton_3d_position_list = []
    for i, file in enumerate(skeleton_parse_result):
        skeleton_data_2d = open(file)
        skeleton_tmp = json.load(skeleton_data_2d)
        skeleton_3d_position = map2d_skeleton_to3D(skeleton_tmp, i, depth_images)
        skeleton_3d_position_list.append(skeleton_3d_position)
        skeleton_data_2d.close()


    'Object positions'
    # objects_file_path = "/home/yuankai/state_of_the_art/har/marsil/model-exploration/src/simulation/kit_bimanual/3d_objects"

    # objects_file_path = "sample_data/3d_objects"

    objects_file_path = f"{DATA_DIR}/3d_objects"
    objects_2d_file_path = f"{DATA_DIR}/2d_objects"

    object_spatial_relations_list = []
    video_transformation_list = []
    object_parse_result = parse_dir(objects_file_path)
    object_2d_parse_result = parse_dir(objects_2d_file_path)

    activity_gt_label = "cooking_with_bowls"

    # vis
    out_point = o3d.geometry.PointCloud()
    vis = o3d.visualization.Visualizer()
    # vis.create_window()
    # vis.add_geometry(out_point)

    # map
    POSE_BODY_25_BODY_PARTS = {
        "Nose": 0,
        "Neck": 1,
        "RShoulder": 2,
        "RElbow": 3,
        "RWrist": 4,
        "LShoulder": 5,
        "LElbow": 6,
        "LWrist": 7,
        "MidHip": 8,
        "RHip": 9,
        "RKnee": 10,
        "RAnkle": 11,
        "LHip": 12,
        "LKnee": 13,
        "LAnkle": 14,
        "REye": 15,
        "LEye": 16,
        "REar": 17,
        "LEar": 18,
        "LBigToe": 19,
        "LSmallToe": 20,
        "LHeel": 21,
        "RBigToe": 22,
        "RSmallToe": 23,
        "RHeel": 24,
        "Background": 25
    }

    "Build the Train Dataset"
    for i, file in enumerate(object_parse_result):
        frame_idx = i
        file_data = open(file)
        data_tmp = json.load(file_data)

        curr_file_2d = object_2d_parse_result[i]
        curr_file_2d_data = open(curr_file_2d)
        curr_data_2d_tmp = json.load(curr_file_2d_data)

        curr_depth_image = depth_images[i]
        curr_rgb_image = rgb_images[i]
        if i > 0:
            past_depth_image = depth_images[i-1]
            past_rgb_images = rgb_images[i-1]
            past_file_2d = object_2d_parse_result[i-1]
            past_file_2d_data = open(past_file_2d)
            past_data_2d_tmp = json.load(past_file_2d_data)
        else:
            past_depth_image = curr_depth_image
            past_rgb_image = curr_rgb_image
            past_file_2d = curr_file_2d
            past_file_2d_data = curr_file_2d_data
            past_data_2d_tmp = curr_data_2d_tmp

        curr_trans = []
        # curr_trans = calculate_trans(curr_data_2d_tmp, past_data_2d_tmp, curr_depth_image, past_depth_image, curr_rgb_image, past_rgb_image)
        # print("curr_trans:", curr_trans)

        righthand_gt_label = righthand_ground_truth_labels[i]
        lefthand_gt_label = lefthand_ground_truth_labels[i]
        objs_dict_tmp = transformation(data_tmp, curr_trans, i, righthand_gt_label, lefthand_gt_label)
        # objs_dict_tmp = transformation(data_tmp, [], i, righthand_gt_label, lefthand_gt_label)
        skeleton_list = skeleton_3d_position_list[i]

        # vis center & joint
        # function vis_cp
        out_point_pf, line_pcd, out_center_pf, out_obj_pf = vis_cp(objs_dict_tmp, skeleton_list)

        axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])

        # vis = o3d.visualization.Visualizer()

        vis.create_window(window_name='Open3D_window', width=600, height=600, left=10, top=30, visible=True)
        # vis.get_render_option().point_size = 10  # set size
        # add_geometry

        vis.clear_geometries() #clear
        vis.add_geometry(axis_pcd)

        # obj center for 3d
        # vis.add_geometry(out_center_pf)
        # obj for 2d
        for idx, obj in enumerate(out_obj_pf):
            vis.add_geometry(obj)
        vis.add_geometry(out_point_pf)  # joint location
        vis.add_geometry(line_pcd)  # line
        # vis.update_geometry()
        vis.poll_events()
        vis.update_renderer()

        frame_dict = {
            "frame_index": frame_idx,
            "ojs_info": objs_dict_tmp,
            "skeleton_info": skeleton_list,
            "righthand_gt_label": righthand_gt_label,
            "lefthand_gt_label": lefthand_gt_label,
            "activity_gt": activity_gt_label
        }
        video_transformation_list.append(frame_dict)
        file_data.close()
    #
    # print(video_transformation_list)

    # skeleton_visulisation(skeleton_3d_position_list)
    #
    # 'Write down the info in .csv file'
    # with open("video_info.csv", 'w') as csv_file:
    #     writer = csv.writer(csv_file)
    #     for key, value in enumerate(video_transformation_list):
    #         writer.writerow(value)
    "Write down the infor in .json"
    # with open("../data.json", 'w') as json_file: default
    with open("data_wt.json", 'w') as json_file:
        # for key, value in enumerate(video_transformation_list):
        json.dump(video_transformation_list, json_file, indent=4)
        print("save data.json processing finished")


    # skeleton_visulisation(skeleton_3d_position_list)

    os.makedirs(f"{DATA_DIR}/result", exist_ok=True)
    'Write down the info in .csv file'
    with open(f"{DATA_DIR}/result/video_info.csv", 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in enumerate(video_transformation_list):
            writer.writerow(value)

