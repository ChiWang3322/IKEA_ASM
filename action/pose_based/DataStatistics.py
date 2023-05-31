import os, logging, math, time, sys, argparse, numpy as np, copy, time, yaml, logging, datetime, shutil, math
from yaml.loader import SafeLoader
from tqdm import tqdm
sys.path.append('../')# for dataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.optim.lr_scheduler as lr_scheduler

from IKEAActionDataset import IKEAPoseActionVideoClipDataset as Dataset

from tools import *
import matplotlib.pyplot as plt

dataset_path='/media/zhihao/Chi_SamSungT7/IKEA_ASM/'
db_filename = 'ikea_annotation_db_full'
train_filename = 'train_cross_env.txt'
test_filename = 'test_cross_env.txt'
test_transforms = None
camera='dev3'
frame_skip=1
frames_per_clip=50
load_mode='img'
pose_relative_path='predictions/pose2d/openpose'
arch='EGCN'
with_obj=True
test_dataset = Dataset(dataset_path, db_filename=db_filename, train_filename=train_filename,
                    test_filename=test_filename, transform=test_transforms, set='train', camera=camera,
                    frame_skip=frame_skip, frames_per_clip=frames_per_clip, mode=load_mode,
                    pose_path=pose_relative_path, arch=arch, with_obj=with_obj)

test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, num_workers=8, shuffle = False,
                                            pin_memory=True)

# (video_full_path, label, n_frames, label_simple), label_simple:(action_id, start frame, end frame)
keys = test_dataset.action_list
# print(keys)
default_value = 0
video_statistic = {key: default_value for key in keys}
action_avg_length = {key: default_value for key in keys}
video_set = test_dataset.video_set
for data in video_set:
    video_full_path, label, n_frames, label_simple = data
    for action_id, start_frame, end_frame in label_simple:
        if action_id == None:
            action = 'NA'
        else:
            action = keys[action_id]

        video_statistic[action] += 1
        action_avg_length[action] += end_frame - start_frame + 1

# print(video_statistic)
label_count = []
avg_length = []
for i in range(len(keys)):
    action = keys[i]
    count = video_statistic[action]
    label_count.append(count)

    total_length = action_avg_length[action]
    avg_length.append(total_length/count)



# create data for the bar plot
colors = ['red', 'green','orange']
x = keys
y1 = np.array(label_count)
y2 = np.array(avg_length)
# print(y)
# create the bar plot

plt.figure(figsize=(6,5))

plt.barh(x, y1)
sorted_indexes = np.argsort(-y1)
for i in range(1):
    plt.barh(x[sorted_indexes[i]], y1[sorted_indexes[i]], color=colors[i])
    plt.text(y1[sorted_indexes[i]]/2, 29, 'spin leg', color='black', fontweight='bold', fontsize=12)

# add labels and title
plt.ylabel('Action')
plt.xlabel('Count')
plt.title('Action Count')


save_path = '../../dataset_statistics/Action_Count.png'
# display the plot
plt.savefig(save_path, dpi=150)

plt.clf()
sorted_indexes = np.argsort(-y2)
plt.barh(x, y2)
for i in range(3):
    plt.barh(x[sorted_indexes[i]], y2[sorted_indexes[i]], color=colors[i])
# add labels and title
plt.ylabel('Action')
plt.xlabel('Average Duration')
plt.title('Action Average Duration')

save_path = '../../dataset_statistics/Action_Average_Duration.png'
# display the plot
plt.savefig(save_path, dpi=150)


