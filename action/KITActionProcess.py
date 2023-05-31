import os
import shutil
from matplotlib import pyplot as plt
import json
import copy
def process_images():
    path = '/media/zhihao/Chi_SamSungT7/KIT_Bimanual/images'

    subjects_list = os.listdir(path)
    subjects_list = ['subject_5', 'subject_6']
    for subject in subjects_list:

        subject_dir = os.path.join(path, subject)
        print(subject_dir)
        task_list = os.listdir(subject_dir)
        for task in task_list:
            task_dir = os.path.join(subject_dir, task)
            print(task_dir)
            take_list = os.listdir(task_dir)
            for take in take_list:
                chunk_dir = os.path.join(task_dir, take, 'rgb')
                print(chunk_dir)
                chunk_list = os.listdir(chunk_dir)
                for chunk in chunk_list:
                    if chunk == 'metadata.csv':
                        continue
                    
                    frame_dir = os.path.join(chunk_dir, chunk)
                    frame_list = os.listdir(frame_dir)
                    for frame in frame_list:
                        source = os.path.join(frame_dir, frame)
                        destination = os.path.join(chunk_dir, frame)
                        print(source)
                        print(destination)
                        shutil.copy2(source, destination)

def KITActionStatistics():
    path = '/media/zhihao/Chi_SamSungT7/KIT_Bimanual/images'
    count = []
    subjects_list = os.listdir(path)
    for subject in subjects_list:
        subject_dir = os.path.join(path, subject)
        print(subject_dir)
        task_list = os.listdir(subject_dir)
        for task in task_list:
            task_dir = os.path.join(subject_dir, task)
            print(task_dir)
            take_list = os.listdir(task_dir)
            for take in take_list:
                chunk_dir = os.path.join(task_dir, take, 'rgb')
                print(chunk_dir)
                chunk_list = os.listdir(chunk_dir)

                count.append(len(chunk_list))
    start = 137
    end = 1047
    num_elements = 360

    element_list = [start + (end - start) * i / (num_elements - 1) for i in range(num_elements)]
    print("max:{}, min:{}".format(max(count), min(count)))
    plt.bar(element_list, count)
    plt.show()
def write_action_list():
    action_list = {0:'idle', 1:'approach', 2:'retreat', 3:'lift', 4:'place', 5:'hold', 6:'pour', 
                    7:'cut', 8:'hammer', 9:'saw', 10:'stir', 11:'screw', 12:'drink', 13:'wipe'}
    path = '/media/zhihao/Chi_SamSungT7/KIT_Bimanual/action_list.json'
    with open(path, 'w') as file:
        json.dump(action_list, file)

def process_objects():


    def get_new_obj_data(object_dir, new_dir, num_frame):
        new_obj_dir = object_dir.replace('2d_objects', '2d_object')
        print('New dir:', new_obj_dir)
        # print('Old dir:', object_dir)
        os.makedirs(new_obj_dir, exist_ok=True)
        # Process frame
        for i in range(num_frame):
            # print('Frame:', i)
            # Read frame data
            frame_path = os.path.join(object_dir, 'frame_' + str(i) + '.json')
            f = open(frame_path)
            data = json.load(f)
            # Store number of objects
            if i == 0:
                num_objects = len(data)
                last_obj_data = []
            obj_class = []
            obj_data = []
            # Store last object class
            last_obj_class = []
            for last_obj in last_obj_data:
                last_obj_class.append(last_obj['class_id'])

            # Filter objects with low certainty
            for obj in data:
                bbox = obj['bounding_box']

                for candidates in obj['candidates']:
                    certainty = candidates['certainty']
                    # Filter out low certainty objects
                    if certainty < 0.9:
                        continue
                    class_id = candidates['class_index']
                    class_color = candidates['colour']
                    bbox['class_id'] = class_id
                    bbox['class_color'] = class_color
                    bbox['certainty'] = certainty
                    obj_class.append(class_id)
                    obj_data.append(bbox)
            if i == 0:
                num_obj = len(obj_data)
            # if len(obj_data) < len(last_obj_data):
            #     obj_data = copy.deepcopy(last_obj_data)
            # last_obj_data = copy.deepcopy(obj_data)

            missing_obj = list(set(last_obj_class).difference(set(obj_class)))
            for m_obj in missing_obj:
                for last_obj in last_obj_data:
                    # print(last_obj)
                    if last_obj['class_id'] == m_obj:
                        obj_data.append(last_obj)
            # print('l clas:', last_obj_class)
            # print('c class:', obj_class)
            # print('---------------------------------------')
            # if len(obj_class) > len(last_obj_class) and i != 0:
            #     obj_data = copy.deepcopy(last_obj_data)
            #     last_obj_data = copy.deepcopy(obj_data)
            #     continue

            # print(len(obj_data))
            # Compensate missing objects
            # print(len(last_obj_data))
            # if len(last_obj_data) != 0:
            #     missing_obj = [i for i in last_obj_class if i not in obj_class]
            #     # print(missing_obj)
            #     # Has missing objects
            #     if len(missing_obj) != 0:
            #         for m_obj in missing_obj:
            #             for last_obj in last_obj_data:
            #                 # print(last_obj)
            #                 if last_obj['class_id'] == m_obj:
            #                     obj_data.append(last_obj)
            # Store new data in new dir
            new_frame_path = os.path.join(new_obj_dir, 'frame_' + str(i) + '.json')
            last_obj_data = copy.deepcopy(obj_data)
            with open(new_frame_path, 'w') as f:
                json.dump(obj_data, f)


    new_dir = '2d_object'
    dataset_path = '/media/zhihao/Chi_SamSungT7/KIT_Bimanual/images'
    subject_list = os.listdir(dataset_path)
    for subject in subject_list:
        subject_dir = os.path.join(dataset_path, subject)
        task_list = os.listdir(subject_dir)
        for task in task_list:
            task_dir = os.path.join(subject_dir, task)
            take_list = os.listdir(task_dir)
            for take in take_list:
                take_dir = os.path.join(task_dir, take) 
                objects_dir = os.path.join(take_dir, '2d_objects')
                # print(objects_dir)
                num_frame = len(os.listdir(objects_dir))
                get_new_obj_data(objects_dir, new_dir, num_frame)

def check_frame_size():
    import pandas as pd
    dataset_path = '/media/zhihao/Chi_SamSungT7/KIT_Bimanual/images'
    subject_list = os.listdir(dataset_path)
    flag = 0
    for subject in subject_list:
        subject_dir = os.path.join(dataset_path, subject)
        task_list = os.listdir(subject_dir)
        for task in task_list:
            task_dir = os.path.join(subject_dir, task)
            take_list = os.listdir(task_dir)
            for take in take_list:
                take_dir = os.path.join(task_dir, take) 
                csv_file = os.path.join(take_dir, 'rgb', 'metadata.csv')
                metadata = pd.read_csv(csv_file)
                metadata = metadata.set_index('name')
                Width = int(metadata.loc['frameWidth',:]['value'])
                Height = int(metadata.loc['frameHeight',:]['value'])
                if Width != 640 or Height != 480:
                    print("New frame size:({}, {})".format(Width, Height))
                    flag = 1
    if flag == 0:
        print('All frame size is (640, 480)...')

def check_total_frame():
    import pandas as pd
    dataset_path = '/media/zhihao/Chi_SamSungT7/KIT_Bimanual/images'
    subject_list = os.listdir(dataset_path)
    flag = 0
    total_rgb_frame = 0
    total_depth_frame = 0
    for subject in subject_list:
        subject_dir = os.path.join(dataset_path, subject)
        task_list = os.listdir(subject_dir)
        for task in task_list:
            task_dir = os.path.join(subject_dir, task)
            take_list = os.listdir(task_dir)
            for take in take_list:
                take_dir = os.path.join(task_dir, take) 
                csv_file = os.path.join(take_dir, 'rgb', 'metadata.csv')
                metadata = pd.read_csv(csv_file)
                metadata = metadata.set_index('name')
                num_frame = int(metadata.loc['frameCount',:]['value'])
                total_rgb_frame += num_frame

                csv_file = os.path.join(take_dir, 'depth', 'metadata.csv')
                metadata = pd.read_csv(csv_file)
                metadata = metadata.set_index('name')
                num_frame = int(metadata.loc['frameCount',:]['value'])
                total_depth_frame += num_frame


    print('Total RGB Frame:', total_rgb_frame)
    print('Total Depth Frame:', total_depth_frame)

def check_pose_confidence():
    import pandas as pd
    from tqdm import tqdm
    dataset_path = '/media/zhihao/Chi_SamSungT7/KIT_Bimanual/images'
    subject_list = os.listdir(dataset_path)
    flag = 0
    total_rgb_frame = 0
    total_depth_frame = 0
    f = open('/media/zhihao/Chi_SamSungT7/KIT_Bimanual/images/subject_1/task_1_k_cooking/take_0/body_pose/frame_20.json')
    pose_data = json.load(f)[0]
    keys = list(pose_data.keys())
    pose_confidence = {}
    for key in keys:
        pose_confidence[key] = [0, 0]

    for subject in subject_list:
        subject_dir = os.path.join(dataset_path, subject)
        task_list = os.listdir(subject_dir)
        for task in tqdm(task_list):
            task_dir = os.path.join(subject_dir, task)
            take_list = os.listdir(task_dir)
            for take in take_list:
                take_dir = os.path.join(task_dir, take) 
                csv_file = os.path.join(take_dir, 'depth', 'metadata.csv')
                metadata = pd.read_csv(csv_file)
                metadata = metadata.set_index('name')
                num_frame = int(metadata.loc['frameCount',:]['value'])
                for i in range(num_frame):
                    pose_path = os.path.join(take_dir, 'body_pose', 'frame_' + str(i) + '.json')
                    with open(pose_path) as file:
                        pose_data = json.load(file)[0]
                    for key in keys:
                        # print('OK')
                        pose_confidence[key][0] += pose_data[key]['confidence']


    avg_confidence = [0] * len(keys)
    for i in range(len(keys)):
        avg_confidence[i] = pose_confidence[keys[i]][0] / 221108
    
        

    plt.bar(keys, avg_confidence)
    plt.xlabel('Categories')
    plt.ylabel('Values')
    plt.title('Bar Plot')

if __name__ == '__main__':
    check_pose_confidence()