# Author: Yizhak Ben-Shabat (Itzik), 2020
# <sitzikbs at gmail dot com>
# train pose based action recognition  methods on IKEA ASM dataset


import os, logging, math, time, sys, argparse, numpy as np, copy, time, yaml, logging, datetime, shutil
from yaml.loader import SafeLoader
from tqdm import tqdm

sys.path.append('../') # pose_based
sys.path.append('../clip_based/i3d/')  # for utils and video transforms
sys.path.append('../..')    #for data loader
sys.path.append('../../..')# for dataset
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
 
import st_gcn, agcn, st2ransformer_dsta
from EfficientGCN.nets import EfficientGCN as EGCN

from Obj_stream import ObjectNet
from net.utils.graph import Graph as g
import i3d_utils as utils


from EfficientGCN.activations import *
from IKEAActionDataset import IKEAPoseActionVideoClipDataset as Dataset
from torch.utils.tensorboard import SummaryWriter






#from torch.utils.tensorboard import SummaryWriter
def init_parser():
    parser = argparse.ArgumentParser(description='Method for Skeleton-based Action Recognition')
    # Settings
    parser.add_argument('--config', type=str, default='', help='path to config')
    parser.add_argument('--load_mode', type=str, default='img', help='dataset loader mode to load videos or images: '
                                                                    'vid | img')
    parser.add_argument('--input_type', type=str, default='rgb', help='depth | rgb ')
    parser.add_argument('--logdir', type=str, default='./log/debug/', help='path to model save dir')
    parser.add_argument('--with_obj', type=bool, default=False, help='With object or not')
    # Dataset
    parser.add_argument('--dataset_path', type=str,
                        default='/home/chiwang/Python/IKEA_Benchmark/IKEA_ASM_Dataset/dataset/ANU_ikea_dataset/', help='path to dataset')
    parser.add_argument('--obj_path', type=str,
                        default='./', help='path to object data')
    parser.add_argument('--db_filename', type=str, default='ikea_annotation_db_full',
                        help='database file name within dataset path')
    parser.add_argument('--camera', type=str, default='dev3', help='dataset camera view: dev1 | dev2 | dev3 ')
    parser.add_argument('--pose_relative_path', type=str, default='predictions/pose2d/openpose',
                        help='path to pose dir within the dataset dir')
    parser.add_argument('--train_filename', type=str, default='train_cross_env.txt',
                        help='path to train filename')
    parser.add_argument('--test_filename', type=str, default='test_cross_env.txt',
                        help='path to test filename')
    # Visualization
    parser.add_argument('--visualize_skeleton', type=bool, default=False, help='visualize skeleton or not')
    parser.add_argument('--visualize_obj', type=bool, default=False, help='visualize object or not')
    parser.add_argument('--visualize_rgb', type=bool, default=False, help='visualize rgb imgae or not')

    # Dataloader
    parser.add_argument('--frame_skip', type=int, default=1, help='reduce fps by skippig frames')
    parser.add_argument('--batch_size', type=int, default=128, help='number of clips per batch')
    parser.add_argument('--frames_per_clip', type=int, default=32, help='number of frames per clip')

    # Training
    parser.add_argument('--n_epochs', type=int, default=3000, help='number of epochs')
    parser.add_argument('--steps_per_update', type=int, default=1, help='number of steps per backprop update')
    parser.add_argument('--refine', type=bool, default=False, help='flag to refine the model')
    parser.add_argument('--refine_epoch', type=int, default=0, help='refine model from this epoch')
    parser.add_argument('--gpu_idx', type=int, default=0, help='set < 0 to use CPU, 999 to use all available gpus')

    # Model
    parser.add_argument('--arch', type=str, default='HCN', help='which architecture to use')
    parser.add_argument('--model_args', default=dict(), help='Args for creating model')

    # Optimizer
    parser.add_argument('--optimizer', type=str, default='', help='Initial optimizer')
    parser.add_argument('--optimizer_args', default=dict(), help='Args for optimizer')

    # Scheduler
    parser.add_argument('--lr_scheduler', '-ls', type=str, default='', help='Initial learning rate scheduler')
    parser.add_argument('--scheduler_args', default=dict(), help='Args for scheduler')
    return parser

def init_model(model_type, model_args, num_classes):
    if model_type == 'STGCN':
        # open-pose layout
        graph_args = {'layout': 'openpose', 'strategy': 'spatial'} 
        model = st_gcn.Model(num_class=num_classes, **model_args)

    elif model_type == 'AGCN':
        model =  agcn.Model(num_class=num_classes, **model_args)

    elif model_type == 'EGCN':
            # Graph and activation function
            graph = g()
            __activations = {
                'relu': nn.ReLU(inplace=True),
                'relu6': nn.ReLU6(inplace=True),
                'hswish': HardSwish(inplace=True),
                'swish': Swish(inplace=True),
            }
            # Rescale block function
            def rescale_block(block_args, scale_args, scale_factor):
                channel_scaler = math.pow(scale_args[0], scale_factor)
                depth_scaler = math.pow(scale_args[1], scale_factor)
                new_block_args = []
                for [channel, stride, depth] in block_args:
                    channel = max(int(round(channel * channel_scaler / 16)) * 16, 16)
                    depth = int(round(depth * depth_scaler))
                    new_block_args.append([channel, int(stride), depth])
                return new_block_args
            # Rescaled block args
            model_args['block_args'] = rescale_block(model_args['block_args'], 
                                        scale_args=model_args['scale_args'],
                                        scale_factor=model_args['scale_factor'])

            print('New block args:', model_args['block_args'])

            kargs = kwargs = {
                'num_class': num_classes,
                'A': graph.A,
                'act': __activations[model_args['act_type']]
            }
            model = EGCN(**model_args, **kargs)
    
    elif model_type == 'DSTA':

        model = st2ransformer_dsta.DSTANet(num_class=num_classes,**model_args)  
    else:
        raise ValueError("Unsupported architecture: please select EGCN | STGCN | AGCN | STGAN")
    return model



def visualize_data():
    pass

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

def batch_multi_input(inputs:torch.Tensor, object_data, with_obj):
    N, C, T, V, M = inputs.size()
    connection = np.array([1,1,1,2,3,1,5,6,2,8,9,5,11,12,0,0,14,15])
    new_input = []
    object_data = object_data.numpy()
    # For over each batch
    for i in range(N): 
        objects = object_data[i]
        joint, velocity, bone = multi_input(inputs[i], connection)
        if with_obj:
            joint = np.concatenate((joint, objects), axis=0)
            velocity = np.concatenate((velocity, objects), axis=0)
            bone = np.concatenate((bone, objects), axis=0)
        
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

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

def run(args):
    # Initialize summary writer with specific logdir
    writer_log = args.logdir.split('/')[2]
    writer_dir = os.path.join('./runs', writer_log + '_split')
    if os.path.exists(writer_dir):
        shutil.rmtree(writer_dir)
        print("Removed existed writer directory......")
    print("Writer dir:", writer_dir)
    writer = SummaryWriter(writer_dir)


    os.makedirs(args.logdir, exist_ok=True)

    # setup dataset
    train_transforms = None
    test_transforms = None
    train_dataset = Dataset(args.dataset_path, db_filename=args.db_filename, train_filename=args.train_filename,
                        test_filename=args.test_filename,
                        transform=train_transforms, set='train', camera=args.camera, frame_skip=args.frame_skip,
                        frames_per_clip=args.frames_per_clip, mode=args.load_mode, pose_path=args.pose_relative_path, 
                        arch=args.arch, obj_path=args.obj_path, with_obj=args.with_obj)
    # set the split ratio
    split_ratio = 0.8

    # split the dataset into training and validation sets
    dataset_size = len(train_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(split_ratio * dataset_size))
    np.random.shuffle(indices)
    train_indices, val_indices = indices[:split], indices[split:]

    # define the samplers for training and validation sets
    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)

    # define the dataloaders
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=8,
                                                  pin_memory=True, sampler=train_sampler)
    test_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, sampler=val_sampler, 
                                                    num_workers=8,pin_memory=True)


    
    # setup the model
    arch = args.arch
    num_classes = train_dataset.num_classes
    model_args = args.model_args

    logging.info("----------------------Dataset INFO----------------------")
    print('Dataset: IKEA_ASM')
    print('Dataset path:', args.dataset_path)
    print('Number of clips in the whole dataset(train_cross_env.txt):', int(len(train_dataset)))
    print("Number of clips in the train dataset:{}".format(int(len(train_dataset)*0.8)))
    print("Number of clips in the test dataset:{}".format(int(len(train_dataset)*0.2)))
    print("Classes:{}\nbatch_size:{}\nframe_skip:{}\nframes_per_clip:{}".format(num_classes, args.batch_size, args.frame_skip, args.frames_per_clip))
    print("Device:{}\ntrain filename:{}\ntest_filename:{}".format(args.camera, args.train_filename, args.test_filename))
    print("Object Included:{}".format(args.with_obj))

    

    # Here use import class easier
    model = init_model(args.arch, args.model_args, num_classes)
   
    ########################################
    ##### Insert code here for refine ######
    ########################################

    if not next(model.parameters()).is_cuda:
        model.cuda()


    # Init optimizer
    optim_args = args.optimizer_args
    optim_type = args.optimizer
    opt_args = optim_args[optim_type]
    optimizer = optim.Adam(model.parameters(), **opt_args)

    # Init learning rate scheduler
    lr_sched = optim.lr_scheduler.MultiStepLR(optimizer, **(args.scheduler_args['MultiStepLR']))


    # Print Optimizer INFO
    logging.info("-----------------Optimizer and SCheduler INFO----------------------")
    print('Using optimizer:{}'.format(optim_type))
    print('LR Scheduler:{}, milestone: {}, gamma:{}'.format('MultiStepLR', 
                                                        args.scheduler_args['MultiStepLR']['milestones'], 
                                                        args.scheduler_args['MultiStepLR']['gamma']))

    model_total_params = sum(p.numel() for p in model.parameters())
    model_total_params = model_total_params / 10**6

    # Print model INFO
    logging.info("-----------------------Model INFO---------------------------------")
    print('Skeleton stream:{}\nnumber of parameters:{}M'.format(arch, round(model_total_params, 2)))
    print('Model args:', model_args)
    
    # train iterations
    train_num_batch = len(train_dataloader)
    test_num_batch = len(test_dataloader)


    best_acc = 0

        
    max_steps = args.n_epochs

    def process_skeleton_data(model_type, skeleton_data, object_data, with_obj):
        # Model type
        # EGCN, batch_multi_input(skeleton_data, object_data)


        if model_type == 'EGCN':
            skeleton_data = batch_multi_input(skeleton_data, object_data, with_obj)
        else:
            if with_obj:
                skeleton_data = append_object_data(skeleton_data, object_data)
            
        inputs = skeleton_data
        

        return inputs
    
    def test_inputs(arch, inputs, skeleton_data, object_data, with_obj):
        inputs = inputs.numpy()
        skeleton_data = skeleton_data.numpy()
        object_data = object_data.numpy()

        # Test inputs shape
        if with_obj:
            N, C_s, T, V, M = skeleton_data.shape
            N, C_o, T, V, M = object_data.shape
            if arch == 'EGCN':
                assert inputs.shape == (N, 3, C_s*2 + C_o, T, V, M), "Inputs shape wrong when with objects"
            else:
                assert inputs.shape == (N, C_s + C_o, T, V, M), "Inputs shape wrong when with objects"
        else:
            N, C_s, T, V, M = skeleton_data.shape
            # print("11111",object_data.shape)
            # N, C_o, T, V, M = object_data.shape
            if arch == 'EGCN':
                assert inputs.shape == (N, 3, C_s*2, T, V, M), "Inputs shape wrong when with objects"
            else:
                assert inputs.shape == (N, C_s, T, V, M), "Inputs shape wrong when without objects"
        
        # Test inputs value
        if with_obj:
            N, C_s, T, V, M = skeleton_data.shape
            N, C_o, T, V, M = object_data.shape
            
            test_Cs = 1
            test_Co = 1
            test_Ci = test_Cs + C_s + test_Co - 1
            test_skeleton_data = skeleton_data[0, test_Cs, 3, 2, 0]
            test_object_data = object_data[0, test_Co, 3, 2, 0]
            if arch =='EGCN':
                pass
            else:
                test_inputs1 = inputs[0, test_Cs, 3, 2, 0]
                test_inputs2 = inputs[0, test_Ci, 3, 2, 0]
            
            
                assert test_inputs1 == test_skeleton_data, "Inputs data wrong when with objects(skeleton)"
                assert test_inputs2 == test_object_data, "Inputs data wrong when with objects(objects)"
        else:
            N, C_s, T, V, M = skeleton_data.shape
            # N, C_o, T, V, M = object_data.shape
            if arch == 'EGCN':
                # Inputs shape: N, B, C, T, V, M
                assert inputs[0, 0, 1, 3, 2, 0] == skeleton_data[0, 1, 3, 2, 0], "Inputs data wrong when without objects"
            else:
                assert inputs[0, 1, 3, 2, 0] == skeleton_data[0, 1, 3, 2, 0], "Inputs data wrong when without objects"

    def test_skeleton_data(skeleton_data):
        pass
        # skeleton_data = skeleton_data.numpy()
        # # if EGCN
        # assert skeleton_data.shape == (args.batch_size, 2, args.frames_per_clip, 18, 1), "Wrong skeleton data shape"
        # if other models
    def test_object_data(object_data, with_obj=args.with_obj):
        # .any() method, False If the array contains only zeros
        contains_none_zero = object_data.numpy().any()

        # Should contain non-zero element
        if with_obj:
            assert contains_none_zero == True, "With object data but object data contains only zero"
        else:
            assert contains_none_zero == False, "Without object data but object data contains none zero"


    #for epoch in range(num_epochs):
    for steps in range(max_steps):
        # Log info
        logging.info("-------------------Training model-------------------")
        print('Step {}/{}, Learning rate:{}'.format(steps, max_steps, lr_sched.get_last_lr()))
        writer.add_scalar("Current lr", torch.tensor(lr_sched.get_last_lr()), steps)
        # Initialization
        train_loss = 0.0
        num_iter = 0
        optimizer.zero_grad()

        # Iterate over data.
        train_acc = []
        
        # Training phase
        for train_batchind, data in enumerate(tqdm(train_dataloader)):
            model.train(True)
            num_iter += 1
            # get the inputs
            skeleton_data, labels, vid_idx, frame_pad, object_data = data
            
            ###################################################################
            test_object_data(object_data, with_obj=args.with_obj)
            test_skeleton_data(skeleton_data)
            ###################################################################

            inputs = process_skeleton_data(arch, skeleton_data, object_data, args.with_obj)

            ###################################################################
            test_inputs(arch, inputs, skeleton_data, object_data, args.with_obj)
            ###################################################################

            inputs = Variable(inputs.cuda(), requires_grad=True)
            # print("Inputs:", inputs.size())
            


            labels = Variable(labels.cuda())
            labels = torch.argmax(labels, dim=1)

            # Score and feature(ignored)
            logits,_ = model(inputs)
            # Length
            t = args.frames_per_clip


            # Interpolate over all frames in the clip
            per_frame_logits = torch.nn.functional.interpolate(logits.unsqueeze(-1), t, mode='linear', align_corners=True)
            # Loss function
            loss = nn.CrossEntropyLoss()(per_frame_logits, labels)
            # Accumulate loss
            train_loss += loss.item()
            loss.backward()
            # Calculate accuracy
            acc = utils.accuracy_v2(torch.argmax(per_frame_logits, dim=1), labels)

            
            train_acc.append(acc.item())
            
            # Update weights 
            optimizer.step()
            optimizer.zero_grad()

        # Evaluation Phase
        logging.info('-------------------Evaluating model-------------------')
        val_acc = []
        val_loss = 0.0
        model.train(False)  # Set model to evaluate mode
        # start_eval_time = time()
        with torch.no_grad():
            for test_batchind, data in enumerate(tqdm(test_dataloader)):
                skeleton_data, labels, vid_idx, frame_pad, object_data = data

                inputs = process_skeleton_data(arch, skeleton_data, object_data, args.with_obj)

                test_inputs(arch, inputs, skeleton_data, object_data, args.with_obj)

                inputs = Variable(inputs.cuda(), requires_grad=False)
                
                labels = Variable(labels.cuda())
                labels = torch.argmax(labels, dim=1)

                logits,_ = model(inputs)
                t = args.frames_per_clip
                per_frame_logits = torch.nn.functional.interpolate(logits.unsqueeze(-1), t, mode='linear', align_corners=True)
                # Interpolate over all frames in the clip
                per_frame_logits = torch.nn.functional.interpolate(logits.unsqueeze(-1), t, mode='linear', align_corners=True)
                # Loss function
                loss = nn.CrossEntropyLoss()(per_frame_logits, labels)
                # Accumulate loss
                val_loss += loss.item()


                # Calculate accuracy
                acc = utils.accuracy_v2(torch.argmax(per_frame_logits, dim=1), labels)

                val_acc.append(acc.item())

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
                        "lr_state_dict": lr_sched.state_dict()}, os.path.join(args.logdir, 'best_classifier.pth'))
        # Report epoch loss and accuracy
        writer.add_scalars("Loss", {'train':torch.tensor(round(train_loss/train_num_batch, 3))}, steps)
        writer.add_scalars("Loss", {'validation':torch.tensor(round(val_loss/test_num_batch, 3))}, steps)
        writer.add_scalars("Accuracy", {'train':torch.tensor(round(np.mean(train_acc), 3))}, steps)

        writer.add_scalars("Accuracy", {'validation':torch.tensor(round(round(np.mean(val_acc), 3)))}, steps)
        writer.add_scalars("Accuracy", {'best':torch.tensor(round(round(best_acc, 3)))}, steps)
        logging.info('-------------Training Epoch Result-------------')
        print('Loss: {}, Accuracy: {}'.format(round(train_loss/train_num_batch, 3), round(np.mean(train_acc), 3)))
        print('-------------Validation Epoch Result-------------')
        print('Loss: {}, Accuracy: {}'.format(round(val_loss/test_num_batch, 3), round(np.mean(val_acc), 3)))
        # steps += 1
        print('Update lr scheduler...')

        lr_sched.step()
    logging.info('--------------------Traing Finished--------------------')
    print("Best accuracy:", best_acc)
    writer.close()



def update_parameters(parser, args):
    config_path = './configs_v1/'
    if os.path.exists(config_path + args.config + '.yaml'):
        with open(config_path + args.config + '.yaml', 'r') as f:
            try:
                yaml_arg = yaml.safe_load(f, Loader=SafeLoader)
            except:
                yaml_arg = yaml.load(f, Loader=SafeLoader)
            default_arg = vars(args)
            for k in yaml_arg.keys():
                if k not in default_arg.keys():
                    raise ValueError('Do NOT exist this parameter {}'.format(k))
            parser.set_defaults(**yaml_arg)
    else:
        raise ValueError('Do NOT exist this file in '+config_path + args.config + '.yaml')
    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level = logging.INFO)
    print('#############################################')
    print('#############################################')
    print('#############################################')
    print('#############   Training Model  #############')
    print('#############################################')
    print('#############################################')
    print('#############################################')
    parser = init_parser()
    args = parser.parse_args()
    args = update_parameters(parser, args)
    #print(args)

    # Set GPU index
    if not args.gpu_idx == 999:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_idx)  # non-functional
        torch.cuda.set_device(0)

    # need to add argparse
    run(args)
