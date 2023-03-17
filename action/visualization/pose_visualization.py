import os
import sys
sys.path.append('../')
sys.path.append('../..')
from IKEAActionDataset import IKEAPoseActionVideoClipDataset as Dataset
import matplotlib.pyplot as plt
import torch
import argparse
from visualization import vis_utils

parser = argparse.ArgumentParser()
parser.add_argument('-db_filename', type=str,
                    default='/home/sitzikbs/Datasets/ANU_ikea_dataset_smaller_video/ikea_annotation_db_full',
                    help='database file')
parser.add_argument('-dataset_path', type=str,
                    default='/home/sitzikbs/Datasets/ANU_ikea_dataset_smaller_video/', help='path to dataset')
parser.add_argument('-output_path', type=str,
                    default='./vis_output/', help='path to dataset')
parser.add_argument('-example', type=int, default=0, help='example index to visualize')
parser.add_argument('-pose_path', type=str, default='', help='Pose path')
args = parser.parse_args()
output_path = args.output_path
os.makedirs(output_path, exist_ok=True)
batch_size = 64
dataset_path = args.dataset_path
db_filename = args.db_filename

# idx = args.example
# dataset = Dataset(dataset_path, db_filename=db_filename, train_filename=train_filename,
#                        transform=None, set='test', camera='dev3', frame_skip=1,
#                        frames_per_clip=64, resize=None)

# video_set = dataset.video_set
def skeleton_visulisation(skeleton):
    # Skeleton data: C x T x V x M
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111)
    for t in range(T):
        X = skeleton[0, t, :, 0]
        Y = skeleton[1, t, :, 0]
        ax.scatter(X, Y)  # plot the point (2,3,4) on the figure
        plt.show()

    # for i, frame_sk in enumerate(skeleton_3d_position_list):


    

if __name__ == '__main__':
    train_dataset = Dataset('/home/zhihao/ChiWang_MA/IKEA_ASM/dataset/', db_filename='ikea_annotation_db_full', train_filename='train_cross_env.txt',
                        test_filename='test_cross_env.txt', set='train', camera='dev3', frame_skip=1,
                        frames_per_clip=100, mode='img', pose_path='predictions/pose2d/openpose', 
                        arch='EGCN', obj_path='/media/zhihao/DataBase/IKEA_ASM/Object_Tracking')
    # Sample according to label distribution

    # weights = utils.make_weights_for_balanced_classes(train_dataset.clip_set, train_dataset.clip_label_count)
    # sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

    sampler = None
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, sampler=sampler,
                                                   num_workers=8, pin_memory=True)
    for train_batchind, data in enumerate(train_dataloader): 
        # N x C x T x V x M
        joint_data, labels, vid_idx, frame_pad, object_data = data
        N, C, T, V, M = joint_data.size()
        for i in range(N):
            one_batch_skeleton = joint_data[i, :, :, :, :].numpy()
            skeleton_visulisation(one_batch_skeleton)
            input("Press Enter to continue...")

