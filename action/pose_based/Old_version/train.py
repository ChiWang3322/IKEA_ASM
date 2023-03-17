# Author: Yizhak Ben-Shabat (Itzik), 2020
# <sitzikbs at gmail dot com>
# train pose based action recognition  methods on IKEA ASM dataset


import os, logging, math, time, sys, argparse, numpy as np, copy, time
from tqdm import tqdm

sys.path.append('../') # for data loader
sys.path.append('../clip_based/i3d/')  # for utils and video transforms

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import HCN, st_gcn, agcn, st2ransformer_dsta
from EfficientGCN.nets import EfficientGCN as EGCN

from net.utils.graph import Graph as g
import i3d_utils as utils


from EfficientGCN.activations import *
from IKEAActionDataset import IKEAPoseActionVideoClipDataset as Dataset


# from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('--frame_skip', type=int, default=1, help='reduce fps by skippig frames')
parser.add_argument('--batch_size', type=int, default=128, help='number of clips per batch')
parser.add_argument('--n_epochs', type=int, default=3000, help='number of epochs')
parser.add_argument('--steps_per_update', type=int, default=1, help='number of steps per backprop update')
parser.add_argument('--frames_per_clip', type=int, default=32, help='number of frames per clip')
parser.add_argument('--db_filename', type=str, default='ikea_annotation_db_full',
                    help='database file name within dataset path')
parser.add_argument('--arch', type=str, default='HCN', help='which architecture to use')
parser.add_argument('--logdir', type=str, default='./log/debug/', help='path to model save dir')
parser.add_argument('--dataset_path', type=str,
                    default='/home/chiwang/Python/IKEA_Benchmark/IKEA_ASM_Dataset/dataset/ANU_ikea_dataset/', help='path to dataset')
parser.add_argument('--load_mode', type=str, default='img', help='dataset loader mode to load videos or images: '
                                                                 'vid | img')
parser.add_argument('--camera', type=str, default='dev3', help='dataset camera view: dev1 | dev2 | dev3 ')
parser.add_argument('--refine', action="store_true", help='flag to refine the model')
parser.add_argument('--refine_epoch', type=int, default=0, help='refine model from this epoch')
parser.add_argument('--gpu_idx', type=int, default=0, help='set < 0 to use CPU, 999 to use all available gpus')
parser.add_argument('--input_type', type=str, default='rgb', help='depth | rgb ')
parser.add_argument('--pose_relative_path', type=str, default='predictions/pose2d/openpose',
                    help='path to pose dir within the dataset dir')
parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight dacay')
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

def batch_multi_input(inputs:torch.Tensor, object_data):
    N, C, T, V, M = inputs.size()
    connection = np.array([1,1,1,2,3,1,5,6,2,8,9,5,11,12,0,0,14,15])
    new_input = []
    object_data = object_data.numpy()
    # Append the object data to the inputs
    for i in range(N): 
        object_tmp = object_data[i]
        joint, velocity, bone = multi_input(inputs[i], connection)
        joint = np.concatenate((joint, object_tmp), axis=0)
        velocity = np.concatenate((velocity, object_tmp), axis=0)
        bone = np.concatenate((bone, object_tmp), axis=0)
        # print(type(joint))
        # print('new joint:', np.shape(joint))
        data = []
        data.append(joint)
        data.append(velocity)
        data.append(bone)

        new_input.append(data)
    inputs = torch.tensor(np.array(new_input, dtype='f'))
    return inputs
def append_object_data(skeleton_data, object_data):

    N, C, T, V, M = skeleton_data.size()
    # print("Skeleton size:", skeleton_data.size())
    # print("Obj size:", object_data.size())
    object_data = object_data.numpy()
    new_input = []
    # Append the object data to the inputs
    for i in range(N): 
        object_tmp = object_data[i]
        new_input.append(np.concatenate((skeleton_data[i].numpy(), object_tmp), axis=0))
        



    inputs = torch.tensor(np.array(new_input, dtype='f'))
    # print("New input size:", inputs.size())
    return inputs


def run(init_lr=0.0001, max_steps=5e3, frames_per_clip=100, dataset_path='/media/sitzikbs/6TB/ANU_ikea_dataset/',
        train_filename='train_cross_env.txt', testset_filename='test_cross_env.txt',
        db_filename='../ikea_dataset_frame_labeler/ikea_annotation_db', logdir='',
        frame_skip=1, batch_size=8, camera='dev3', refine=False, refine_epoch=0, load_mode='img',
         pose_path='predictions/pose2d/openpose', arch='HCN', steps_per_update=1, 
         momentum=0.9, weight_decay=0.0001, num_frame = 64):


    os.makedirs(logdir, exist_ok=True)

    # setup dataset
    train_transforms = None
    test_transforms = None
    train_dataset = Dataset(dataset_path, db_filename=db_filename, train_filename=train_filename,
                 transform=train_transforms, set='train', camera=camera, frame_skip=frame_skip,
                            frames_per_clip=frames_per_clip, mode=load_mode, pose_path=pose_path, arch=arch)
    print('Dataset: IKEA_ASM')
    print("Number of clips in the dataset:{}".format(len(train_dataset)))
    weights = utils.make_weights_for_balanced_classes(train_dataset.clip_set, train_dataset.clip_label_count)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=sampler,
                                                   num_workers=6, pin_memory=False)

    test_dataset = Dataset(dataset_path, db_filename=db_filename, train_filename=train_filename,
                           test_filename=testset_filename, transform=test_transforms, set='test', camera=camera,
                           frame_skip=frame_skip, frames_per_clip=frames_per_clip, mode=load_mode,
                           pose_path=pose_path, arch=arch)

    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=6,
                                                  pin_memory=False)

    # setup the model
    num_classes = train_dataset.num_classes
    print("Number of classes:", num_classes)
    if arch == 'HCN':
        model = HCN.HCN(in_channel=2, num_joint=19, num_person=1, out_channel=64, window_size=frames_per_clip,
                        num_class=num_classes)
    elif arch == 'STGCN':
        graph_args = {'layout': 'openpose', 'strategy': 'spatial'} #ntu-rgb+d
        model = st_gcn.Model(in_channels=2, num_class=num_classes, graph_args=graph_args,
                             edge_importance_weighting=True, dropout=0.5)
    elif arch == 'AGCN':
        model = agcn.Model(num_class=num_classes, num_point=18, num_person=1, 
                          graph='graph.kinetics.Graph', graph_args={'labeling_mode':'spatial'}, in_channels=7)
    elif arch =='STGAN':
        config = [ [64, 64, 16, 1], [64, 64, 16, 1],
            [64, 128, 32, 2], [128, 128, 32, 1],
            [128, 256, 64, 2], [256, 256, 64, 1],
            [256, 256, 64, 1], [256, 256, 64, 1],
        ]
        model = st2ransformer_dsta.DSTANet(num_class=num_classes, num_point=18, num_frame=num_frame, num_subset=4, dropout=0., config=config, num_person=1,
                 num_channel=2)                      
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
                                    scale_factor=5)
        print('New block args:', block_args)
        act_type = 'swish'
        model = EGCN(data_shape=(3, 9, frames_per_clip, 18, 1),stem_channel = 64,
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
        raise ValueError("Unsupported architecture: please select HCN | ST_GCN | AGCN | STGAN")

    if refine:
        if refine_epoch == 0:
            # raise ValueError("You set the refine epoch to 0. No need to refine, just retrain.")
            print('Refining model')
            refine_model_filename = os.path.join(logdir, 'best_classifier.pth')
            checkpoint = torch.load(refine_model_filename)
            model.load_state_dict(checkpoint["model_state_dict"])
        # refine_model_filename = os.path.join(logdir, str(refine_epoch).zfill(6)+'.pt')
        # checkpoint = torch.load(refine_model_filename)
        # model.load_state_dict(checkpoint["model_state_dict"])


    model.cuda()
    # model = nn.DataParallel(model)
    # Load optimizer
    milestones=[10, 20, 30, 40, 50, 60, 70, 80, 90]
    gamma = 0.6
    if arch == 'STGCN':

        lr = init_lr
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1E-6)
        lr_sched = optim.lr_scheduler.MultiStepLR(optimizer, milestones = milestones, gamma = gamma)
        # criterion = nn.CrossEntropyLoss()  # standard crossentropy loss for classification
    elif arch == 'AGCN':

        optimizer = optim.Adam(
            model.parameters(),
            lr=init_lr,
            weight_decay=weight_decay)
        lr_sched = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones = milestones, gamma = gamma)
    elif arch == 'STGAN':

        optimizer = optim.Adam(model.parameters(), lr=init_lr, weight_decay=weight_decay)
        lr_sched = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones = milestones, gamma = gamma)
    elif arch == 'EGCN':

        optimizer = optim.Adam(model.parameters(), 
                                lr=init_lr, 
                                weight_decay=weight_decay, 
                                betas=[0.9,0.99])
        lr_sched = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones = milestones, gamma = gamma)
    if refine:
        lr_sched.load_state_dict(checkpoint["lr_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    print('Using optimizer for '.format(arch))
    print('LR Scheduler:{}, milestone: {}, gamma:{}'.format('MultiStepLR', milestones, gamma))
    # train_writer = SummaryWriter(os.path.join(logdir, 'train'))
    # test_writer = SummaryWriter(os.path.join(logdir, 'test'))

    model_total_params = sum(p.numel() for p in model.parameters())
    model_total_params = model_total_params / 10**6
    print('Model {}, number of parameters:{}M'.format(arch, round(model_total_params, 2)))

    num_steps_per_update = steps_per_update  # accum gradient - try to have number of examples per update match original code 8*5*4
    eval_steps  = 5
    # train it
    n_examples = 0
    train_num_batch = len(train_dataloader)
    test_num_batch = len(test_dataloader)
    refine_flag = True

    best_acc = 0
    steps = 0
        

    while steps < max_steps:#for epoch in range(num_epochs):
        print('Step {}/{}'.format(steps, max_steps))
        print('Current learning rate:',lr_sched.get_last_lr())
        if steps <= refine_epoch and refine and refine_flag:
            lr_sched.step()
            steps += 1
            n_examples += len(train_dataset.clip_set)
            continue
        else:
            refine_flag = False
        # Each epoch has a training and validation phase

        test_batchind = -1
        test_fraction_done = 0.0
        test_enum = enumerate(test_dataloader, 0)
        tot_loss = 0.0
        num_iter = 0
        optimizer.zero_grad()
        running_tloss = 0
        running_vloss = 0
        # Iterate over data.
        train_acc = []
        val_acc = []
        for train_batchind, data in enumerate(tqdm(train_dataloader)):
            
            num_iter += 1
            # get the inputs
            joint_data, labels, vid_idx, frame_pad, object_data = data
            # print('joint:', joint_data.size())
            # print('object:', object_data.size())
            if arch == 'EGCN':

                inputs = batch_multi_input(joint_data, object_data)
            else:
                inputs = append_object_data(joint_data, object_data)
            # wrap them in Variable
            # N x B x C x T x V x M
            inputs = Variable(inputs.cuda(), requires_grad=True)

            labels = Variable(labels.cuda())
            labels = torch.argmax(labels, dim=1)
            t1 = time.time()

            logits, _ = model(inputs)

            t = frames_per_clip
            # print('Forward:', time.time() - t1)
            per_frame_logits = torch.nn.functional.interpolate(logits.unsqueeze(-1), t, mode='linear', align_corners=True)
            #probs = torch.nn.functional.softmax(per_frame_logits, dim=1)
            loss = nn.CrossEntropyLoss()(per_frame_logits, labels)

            tot_loss += loss.item()
            running_tloss += loss.item()
            loss.backward()

            acc = utils.accuracy_v2(torch.argmax(per_frame_logits, dim=1), labels)

            train_acc.append(acc.item())
            train_fraction_done = (train_batchind + 1) / train_num_batch

            if (num_iter == num_steps_per_update or train_batchind == len(train_dataloader)-1) :
                n_steps = num_steps_per_update
                if train_batchind == len(train_dataloader)-1:
                    n_steps = num_iter
                n_examples += batch_size*n_steps


                optimizer.step()
                optimizer.zero_grad()

                num_iter = 0
                tot_loss = 0.

            if test_fraction_done <= train_fraction_done and test_batchind + 1 < test_num_batch:
            #    print('----------Evaluation----------')
                model.train(False)  # Set model to evaluate mode
                test_batchind, data = next(test_enum)
                inputs, labels, vid_idx, frame_pad, object_data = data
                if arch == 'EGCN':
                    #print('Generating joint, velocity, bone data...')
                    inputs = batch_multi_input(inputs, object_data)
                    #print('Generating new information successfully!------------')
                else:
                    inputs = append_object_data(inputs, object_data)
                # wrap them in Variable
                inputs = Variable(inputs.cuda(), requires_grad=True)
                labels = Variable(labels.cuda())
                labels = torch.argmax(labels, dim=1)

                with torch.no_grad():

                    logits, _ = model(inputs)

                    
                    t = frames_per_clip
                    per_frame_logits = torch.nn.functional.interpolate(logits.unsqueeze(-1), t, mode='linear',
                                                                       align_corners=True)
                    probs = torch.nn.functional.softmax(per_frame_logits, dim=1)

                    loss = nn.CrossEntropyLoss()(per_frame_logits, labels)
                    running_vloss += loss.item()
                    acc = utils.accuracy_v2(torch.argmax(per_frame_logits, dim=1), labels)
                    val_acc.append(acc.item())

                # test_writer.add_scalar('loss', loss.item(), n_examples)
                # test_writer.add_scalar('Accuracy', acc.item(), n_examples)
                test_fraction_done = (test_batchind + 1) / test_num_batch
                model.train(True)
        if steps % 50 == 0:
            # save model
            torch.save({"model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "lr_state_dict": lr_sched.state_dict()},
                       logdir + str(steps).zfill(6) + '.pt')

        # remember best prec@1 and save checkpoint
        epoc_val_acc = np.mean(val_acc)
        is_best = epoc_val_acc > best_acc
        best_acc = max(epoc_val_acc, best_acc)
        if (is_best):
            print('Best accuracy achieved, storing the model...')
            print('Best val_acc: {}'.format(best_acc))
            model_tmp = copy.deepcopy(model.state_dict())
            model.load_state_dict(model_tmp)
            torch.save({"model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "lr_state_dict": lr_sched.state_dict()}, os.path.join(logdir, 'best_classifier.pth'))
        # Report epoch loss and accuracy
        print('Training----------------')
        print('Loss: {}, Accuracy: {}'.format(round(running_tloss/train_num_batch, 3), round(np.mean(train_acc), 3)))
        print('Validation--------------')
        print('Loss: {}, Accuracy: {}'.format(round(running_vloss/test_num_batch, 3), round(np.mean(val_acc), 3)))
        steps += 1
        print('Update lr scheduler...')
        lr_sched.step()
    logging.info("------Training Finished------")
    print("Best accuracy:", best_acc)
    # train_writer.close()
    # test_writer.close()




if __name__ == '__main__':

    if not args.gpu_idx == 999:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_idx)  # non-functional
        torch.cuda.set_device(0)

    # need to add argparse
    print("Starting training ...")
    print("Using data from {}".format(args.camera))
    run(dataset_path=args.dataset_path, logdir=args.logdir,
        frame_skip=args.frame_skip, db_filename=args.db_filename, batch_size=args.batch_size, max_steps=args.n_epochs, camera=args.camera,
        refine=args.refine, refine_epoch=args.refine_epoch, load_mode=args.load_mode, pose_path=args.pose_relative_path,
        arch=args.arch, frames_per_clip=args.frames_per_clip, steps_per_update=args.steps_per_update,
        init_lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, num_frame = args.frames_per_clip)