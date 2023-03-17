import os
import sys
sys.path.append('../')
from IKEAActionDataset import IKEAActionVideoClipDataset as Dataset

import matplotlib.pyplot as plt
import numpy as np


## Parser insert video path, object path, with skeleton, bbox or not
parser = argparse.ArgumentParser()
parser.add_argument('-db_filename', type=str,
                    default='ikea_annotation_db_full',
                    help='database file')
parser.add_argument('-dataset_path', type=str,
                    default='/home/sitzikbs/Datasets/ANU_ikea_dataset_smaller_video/', help='path to dataset')
parser.add_argument('-output_path', type=str,
                    default='./vis_output/', help='path to dataset')
parser.add_argument('-example', type=int, default=0, help='example index to visualize')
args = parser.parse_args()

## Visualize video

## If skeleton 
# Generate random 2D skeleton data

###########################
skeleton_data = []
###########################

# Plot the skeleton data as a scatter plot
plt.scatter(skeleton_data[:, 0], skeleton_data[:, 1])

## If bbox
# Read bbox

# Plot bbox

## Show action text





