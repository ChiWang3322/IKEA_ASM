# Author: Yizhak Ben-Shabat (Itzik), 2020
# <sitzikbs at gmail dot com>
# test pose based action recognition methods on IKEA ASM dataset

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='0'
import argparse
import itertools
import sys
sys.path.append('../') # for data loader
sys.path.append('../clip_based/i3d/')  # for utils and video transforms
import i3d_utils
import utils
import torch
from torch.autograd import Variable
from torchvision import transforms
import videotransforms
import numpy as np
import HCN
import st_gcn
import agcn
from tqdm import tqdm
from IKEAActionDataset import IKEAPoseActionVideoClipDataset as Dataset
from IKEAActionDataset import IKEAActionSegmentDataset as Dataset2
from pyrutils import metrics
from EfficientGCN.nets import EfficientGCN as EGCN
from EfficientGCN.activations import *
from net.utils.graph import Graph as g
import math

parser = argparse.ArgumentParser()
parser.add_argument('--frame_skip', type=int, default=1, help='reduce fps by skipping frames')
parser.add_argument('--batch_size', type=int, default=128, help='number of clips per batch')
parser.add_argument('--frames_per_clip', type=int, default=128, help='number of frames in a sequence')
parser.add_argument('--arch', type=str, default='HCN', help='ST_GCN | HCN indicates which architecture to use')
parser.add_argument('--db_filename', type=str,
                    default='ikea_annotation_db_full',
                    help='database file')
parser.add_argument('--model_path', type=str, default='./log/HCN_128/',
                    help='path to model save dir')
parser.add_argument('--device', default='dev3', help='which camera to load')
parser.add_argument('--model', type=str, default='best_classifier.pth',
                    help='path to model save dir')
parser.add_argument('--dataset_path', type=str,
                    default='/home/chiwang/Python/IKEA_Benchmark/IKEA_ASM_Dataset/dataset/ANU_ikea_dataset/', help='path to dataset')
parser.add_argument('--pose_relative_path', type=str, default='predictions/pose2d/openpose',
                    help='path to pose dir within the dataset dir')
args = parser.parse_args()

def multi_input(data, connection):
    """
    A function used to generate joint, bone and velocity from only joint data

    Inputs
    ----------
    data : C x T x V tensor
        N batch size, C number of channels, T number of frames, V number of joints
    connecion: 1 x V list
        describe the connected point of each joint
    Outputs
    ----------
    joint, velocity and bone data, each of size C*2 x T x V
    """
    # Channels, T, Num joints, num person
    C, T, V, M = data.size()
    data = data.numpy()
    joint = np.zeros((C*2, T, V, M))
    velocity = np.zeros((C*2, T, V, M))
    bone = np.zeros((C*2, T, V, M))
    joint[:C,:,:,:] = data
    for i in range(V):
        # joint_i - center joint(number 1)
        joint[C:,:,i,:] = data[:,:,i,:] - data[:,:,1,:]
    for i in range(T-2):
        velocity[:C,i,:,:] = data[:,i+1,:,:] - data[:,i,:,:]
        velocity[C:,i,:,:] = data[:,i+2,:,:] - data[:,i,:,:]
    for i in range(len(connection)):
        bone[:C,:,i,:] = data[:,:,i,:] - data[:,:,connection[i],:]
    bone_length = 0
    for i in range(C):
        bone_length += bone[i,:,:,:] ** 2
    bone_length = np.sqrt(bone_length) + 0.0001
    for i in range(C):
        bone[C+i,:,:,:] = np.arccos(bone[i,:,:,:] / bone_length)
    return joint, velocity, bone

def batch_multi_input(inputs:torch.Tensor):
    N, C, T, V, M = inputs.size()
    connection = np.array([1,1,1,2,3,1,5,6,2,8,9,5,11,12,0,0,14,15])
    new_input = []
    for i in range(N): 
        joint, velocity, bone = multi_input(inputs[i], connection)
        data = []
        data.append(joint)
        data.append(velocity)
        data.append(bone)
        new_input.append(data)
    inputs = torch.tensor(np.array(new_input, dtype='f'))
    return inputs



def run(dataset_path, db_filename, model_path, output_path, frames_per_clip=16,
        testset_filename='test_cross_env.txt', trainset_filename='train_cross_env.txt', frame_skip=1,
        batch_size=8, device='dev3', arch='HCN', pose_path='predictions/pose2d/openpose'):

    pred_output_filename = os.path.join(output_path, 'pred.npy')
    json_output_filename = os.path.join(output_path, 'action_segments.json')

    # setup dataset
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])

    test_dataset = Dataset(dataset_path, db_filename=db_filename, test_filename=testset_filename,
                           train_filename=trainset_filename, transform=test_transforms, set='test', camera=device,
                           frame_skip=frame_skip, frames_per_clip=frames_per_clip, mode='img', pose_path=pose_path,
                           arch=arch)

    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=6,
                                                  pin_memory=True)
    class_names = utils.squeeze_class_names(test_dataset.action_list)
    print("class names:", class_names)
    # setup the model
    num_classes = test_dataset.num_classes
    print('Num classes:', num_classes)
    if arch == 'HCN':
        model = HCN.HCN(in_channel=2, num_joint=19, num_person=1, out_channel=64, window_size=frames_per_clip,
                        num_class=num_classes)
    elif arch == 'ST_GCN':
        graph_args = {'layout': 'openpose', 'strategy': 'spatial'}  # layout:'ntu-rgb+d'
        model = st_gcn.Model(in_channels=2, num_class=num_classes, graph_args=graph_args,
                             edge_importance_weighting=True, dropout=0.5)
    elif arch == 'AGCN':
        model = agcn.Model(num_class=num_classes, num_point=18, num_person=1, 
                          graph='graph.kinetics.Graph', graph_args={'labeling_mode':'spatial'}, in_channels=2)
    elif arch == 'EGCN':
        #B4
        graph = g()
        __activations = {
            'relu': nn.ReLU(inplace=True),
            'relu6': nn.ReLU6(inplace=True),
            'hswish': HardSwish(inplace=True),
            'swish': Swish(inplace=True),
        }
        def rescale_block(block_args, scale_args, scale_factor):
            channel_scaler = math.pow(scale_args[0], scale_factor)
            depth_scaler = math.pow(scale_args[1], scale_factor)
            new_block_args = []
            for [channel, stride, depth] in block_args:
                channel = max(int(round(channel * channel_scaler / 16)) * 16, 16)
                depth = int(round(depth * depth_scaler))
                new_block_args.append([channel, stride, depth])
            return new_block_args
        block_args = rescale_block([[48,1,0.5],[24,1,0.5],[64,2,1],[128,2,1]], 
                                    scale_args=[1.2,1.35],
                                    scale_factor=4)
        print('New block args:', block_args)
        act_type = 'swish'
        model = EGCN(data_shape=(3, 4, frames_per_clip, 18, 1),stem_channel = 64,
                block_args = block_args,
                fusion_stage = 2,
                num_class = num_classes,
                act =  __activations[act_type],
                att_type =  'stja',
                layer_type = 'Sep',
                drop_prob = 0.25,
                kernel_size = [5,2],
                # scale_args = [1.2,1.35],
                expand_ratio = 2,
                reduct_ratio = 4,
                bias = True,
                edge = True,
                A = graph.A)
    else:
        raise ValueError("Unsupported architecture: please select HCN | ST_GCN")

    checkpoints = torch.load(model_path)
    model.load_state_dict(checkpoints["model_state_dict"]) # load trained model
    model.cuda()
    # model = nn.DataParallel(model)

    n_examples = 0

    # Iterate over data.
    avg_acc = []
    # predicted labels of every test video in one list
    pred_labels_per_video = [[] for i in range(len(test_dataset.video_list))]
    true_labels_per_video = [[] for i in range(len(test_dataset.video_list))]
    logits_per_video = [[] for i in range(len(test_dataset.video_list))]
    f1_10, f1_25, f1_50 = 0, 0, 0
    for test_batchind, data in enumerate(tqdm(test_dataloader)):
        model.train(False)
        # get the inputs
        # labels in one-hot encoding
        inputs, labels, vid_idx, frame_pad = data
        # print('video index:', vid_idx)
        # print('frame pad:', frame_pad)
        # print('size of label:', labels.size())
        # print('inputs size:',inputs.size())
        # print("test_batch index:",test_batchind)
        if arch == 'EGCN':
            inputs = batch_multi_input(inputs)

        # wrap them in Variable
        inputs = Variable(inputs.cuda(), requires_grad=True)
        labels = Variable(labels.cuda())

        t = inputs.size(2)
        if arch == 'EGCN':
            t = inputs.size(3)
        if arch == 'EGCN':
            logits, _ = model(inputs)
        else:
            logits = model(inputs)
        # print("size of model output:", logits.size())
        logits = torch.nn.functional.interpolate(logits.unsqueeze(-1), t, mode='linear', align_corners=True)
        # logits = F.interpolate(logits, t, mode='linear', align_corners=True)  # b x classes x frames
        # print(logits[0])
        acc = i3d_utils.accuracy_v2(torch.argmax(logits, dim=1), torch.argmax(labels, dim=1))
        # y_true = torch.argmax(labels, dim=1).cpu()
        # y_pred = torch.argmax(logits, dim=1).cpu()
        avg_acc.append(acc.item())
        #print('batch Acc: {}, [{} / {}]'.format(acc.item(), test_batchind, len(test_dataloader)))
        logits = logits.permute(0, 2, 1)  # [ batch, classes, frames] -> [ batch, frames, classes]
        # frames_per_clip = inputs.size()[2]
        # print('frames_per_segment:', frames_per_clip)
        logits = logits.reshape(inputs.shape[0] * frames_per_clip, -1)
        labels = labels.reshape(inputs.shape[0] * frames_per_clip, -1)
        pred_labels = torch.argmax(logits, 1).detach().cpu().numpy().tolist()
        true_labels = torch.argmax(labels, 1).detach().cpu().numpy().tolist()
        # print('pred_labels:',np.shape(pred_labels))
        # print('true_labels:',np.shape(true_labels))
        logits = torch.nn.functional.softmax(logits, dim=1).detach().cpu().numpy().tolist()
        # frame_pad = torch.tensor([0])
        pred_labels_per_video, logits_per_video = \
            utils.accume_per_video_predictions(vid_idx, frame_pad,pred_labels_per_video, logits_per_video, pred_labels,
                                     logits, frames_per_clip)
        true_labels_per_video, true_logits_per_video = \
            utils.accume_per_video_predictions(vid_idx, frame_pad,true_labels_per_video, labels.detach().cpu().numpy().tolist(), true_labels,
                                        logits, frames_per_clip)

    f1_10 = metrics.f1_at_k(true_labels_per_video, pred_labels_per_video, num_classes=num_classes, overlap=0.1)
    f1_25 = metrics.f1_at_k(true_labels_per_video, pred_labels_per_video, num_classes=num_classes, overlap=0.25)
    f1_50 = metrics.f1_at_k(true_labels_per_video, pred_labels_per_video, num_classes=num_classes, overlap=0.5)
    
    pred_labels_per_video = [np.array(pred_video_labels) for pred_video_labels in pred_labels_per_video]
    logits_per_video = [np.array(pred_video_logits) for pred_video_logits in logits_per_video]
    true_labels_per_video = [np.array(true_video_labels) for true_video_labels in true_labels_per_video]
    # for true, pred, logits in zip(true_labels_per_video, pred_labels_per_video, logits_per_video):
    #     print('True length:', len(true))
    #     print('Pred length:', len(pred))
    #     print('Logits length:',np.shape(logits))
    # with open('test_true.npy', 'wb') as f:
    #     np.save('test_true', true_labels_per_video)
    # with open('test_pred.npy', 'wb') as f:
    #     np.save('test_pred', pred_labels_per_video)
    np.save(pred_output_filename, {'pred_labels': pred_labels_per_video,
                                   'logits': logits_per_video})
    utils.convert_frame_logits_to_segment_json(logits_per_video, json_output_filename, test_dataset.video_list,
                                               test_dataset.action_list)
    print('Average accuracy(framewise:', np.mean(avg_acc))
    print("#################")
    print("F1@10%:", f1_10)
    print("#################")
    print("F1@25%:", f1_25)
    print("#################")
    print("F1@50%:", f1_50)
    # print("#################")
    # print('F1 macro:',metrics.f1(true_labels_per_video, pred_labels_per_video))

if __name__ == '__main__':
    # need to add argparse
    output_path = os.path.join(args.model_path, 'results')
    os.makedirs(output_path, exist_ok=True)
    # best_classifier.pth
    model_path = os.path.join(args.model_path, args.model)
    run(dataset_path=args.dataset_path, db_filename=args.db_filename, model_path=model_path,
        output_path=output_path, frame_skip=args.frame_skip,  batch_size=args.batch_size,
        device=args.device, arch=args.arch, pose_path=args.pose_relative_path, frames_per_clip=args.frames_per_clip)
    os.system('python3 ../evaluation/evaluate.py --results_path {} --mode vid'.format(output_path))
