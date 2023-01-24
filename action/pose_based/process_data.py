import json
import os
from tqdm import tqdm
import numpy as np


root_directory = 'test'
process_directory = 'test_processed'

# root_directory = 'train'
# process_directory = 'train_processed'
# Make directory
os.makedirs(process_directory, exist_ok=True)
# Make env_dir
env_directory = os.listdir(root_directory)
for env in env_directory:
    new_dir = os.path.join(process_directory, env)
    os.makedirs(new_dir, exist_ok=True)
print(env_directory)
category_count = np.zeros([20])

print('Start processing data')
for env in env_directory:
    item_directory = os.listdir(os.path.join(root_directory, env))

    print('INFO:Load data in {}-{}'.format(root_directory, env))
    for item in tqdm(item_directory):
        # make dir for item directory
        new_data = []
        new_dir = os.path.join(process_directory, env, item, 'dev3')
        os.makedirs(new_dir, exist_ok=True)
        json_directory = os.path.join(root_directory, env, item, 'dev3', 'all_gt_coco_format.json')
        f = open(json_directory)
        data = json.load(f)
        # Check length
        # if data['annotations'][-1]['image_id'] == len(data['images']):
        #     print('Length correct!')
        # else:
        #     print('Length incorrect! Skip')
        #     continue
        for i in range(len(data['annotations'])):

            old_one = data['annotations'][i]    #dict
            # print(old_one.keys())
            # print(old_one['part_id'])
            image_path = data['images'][old_one['image_id'] - 1]['file_name']
            frame_ind = int(image_path.split('.')[0].split('/')[-1])
            # print('frame_ind:', frame_ind)
            new_one = {'category_id': old_one['category_id'],
                     'image_id': old_one['image_id'], 
                     'area': old_one['area'], 
                     'bbox': old_one['bbox'], 
                     'iscrowd': old_one['iscrowd'], 

                     'frame_index': frame_ind,
                     }

            current_path = os.path.join(new_dir, str(frame_ind)+'.json')
            if os.path.exists(current_path):
                f = open(current_path)
                objects = json.load(f)
                objects.append(new_one)
                with open(current_path, 'w') as f:
                    json.dump(objects, f)
            else:
                new_one = [new_one]
                with open(current_path, 'w') as f:
                    json.dump(new_one, f)

        


        



