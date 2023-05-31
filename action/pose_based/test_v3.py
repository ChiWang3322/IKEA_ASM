# Author: Yizhak Ben-Shabat (Itzik), 2020
# <sitzikbs at gmail dot com>
# train pose based action recognition  methods on IKEA ASM dataset


import os, logging, math, time, sys, argparse, numpy as np, copy, time, yaml, logging
from yaml.loader import SafeLoader
from tqdm import tqdm
from thop import profile

sys.path.append('../') # pose_based
sys.path.append('../clip_based/i3d/')  # for utils and video transforms
sys.path.append('../..')    #for data loader
sys.path.append('../../..')# for dataset
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
import i3d_utils
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from pyrutils import metrics
import st_gcn, agcn, st2ransformer_dsta, agcn
from EfficientGCN.nets import EfficientGCN as EGCN
from Obj_stream import ObjectNet
# from net.utils.graph import Graph as g
import utils
from sklearn.metrics import accuracy_score

from EfficientGCN.activations import *
from IKEAActionDataset import IKEAPoseActionVideoClipDataset as Dataset
from tools import *


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
        from net.utils.graph import Graph as g
        # Graph and activation function
        layout = model_args['layout']
        graph = g(layout=layout)
        print("graph.A.shape:",graph.A.shape)
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
    
    # elif model_type == 'DSTA':

    #     model = st2ransformer_dsta.DSTANet(num_class=num_classes,**model_args)  
    else:
        raise ValueError("Unsupported architecture: please select EGCN | STGCN | AGCN | STGAN")
    return model


def run(args):

    os.makedirs(args.logdir, exist_ok=True)

    # setup test dataset
    test_transforms = None
    test_dataset = Dataset(args.dataset_path, db_filename=args.db_filename, train_filename=args.train_filename,
                           test_filename="train_cross_env_small.txt", transform=test_transforms, set='test', camera=args.camera,
                           frame_skip=args.frame_skip, frames_per_clip=args.frames_per_clip, mode=args.load_mode,
                           pose_path=args.pose_relative_path, arch=args.arch, with_obj=args.with_obj)

    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=8, shuffle = False,
                                                  pin_memory=True)
    
    # setup the model
    arch = args.arch
    num_classes = test_dataset.num_classes
    model_args = args.model_args
    print('#############################################')
    print('#############################################')
    print('#############################################')
    print('#############   Testing Model   #############')
    print('#############################################')
    print('#############################################')
    print('#############################################')

    print("\n"+"=========="*8 + "Dataset INFO\n")
    print('Dataset: IKEA_ASM')
    print('Dataset path:', args.dataset_path)
    print("Number of clips in the test dataset:{}".format(len(test_dataset)))
    print("Classes:{}\nbatch_size:{}\nframe_skip:{}\nframes_per_clip:{}".format(num_classes, args.batch_size, args.frame_skip, args.frames_per_clip))
    print("Device:{}\ntrain filename:{}\ntest_filename:{}".format(args.camera, args.train_filename, args.test_filename))
    print("Object Included:{}".format(args.with_obj))
    print("Config file:", args.config)

    # Here use import class easier
    model = init_model(args.arch, args.model_args, num_classes)
    # input_tmp = torch.randn((1, 3, 5, 50, 24, 1))
    # # input_tmp = Variable(input_tmp.cuda(), requires_grad=False)

    # flops, params = profile(model, inputs=(input_tmp,))
    # print('FLOPs = ' + str(flops/1000**3) + 'G')
    model_path = os.path.join(args.logdir, 'best_classifier.pth')
    checkpoints = torch.load(model_path)
    model.load_state_dict(checkpoints["model_state_dict"]) # load trained model
    model.cuda()
    
    ########################################

    ########################################
    ##### Insert code here for refine ######
    ########################################

    if not next(model.parameters()).is_cuda:
        model.cuda()


    model_total_params = sum(p.numel() for p in model.parameters())
    model_total_params = model_total_params / 10**6

    # Print model INFO
    print("\n"+"=========="*8 + "Model INFO\n")
    print('Model:{}\nnumber of parameters:{}M'.format(args.arch, round(model_total_params, 2)))
    print('Model args:', args.model_args)

    # Initialization
    # Iterate over data.
    avg_acc = []
    pred_labels_per_video = [[] for i in range(len(test_dataset.video_list))]
    true_labels_per_video = [[] for i in range(len(test_dataset.video_list))]
    intensity_per_video = [[] for i in range(len(test_dataset.video_list))]
    logits_per_video = [[] for i in range(len(test_dataset.video_list))]
    f1_10, f1_25, f1_50 = 0, 0, 0
    frames_per_clip = args.frames_per_clip
    true = []
    pred = []
    # Training phase
    with torch.no_grad():
        loop = tqdm(enumerate(test_dataloader), total =len(test_dataloader), file = sys.stdout)
        for _, data in loop:
            model.train(False)
            # get the inputs
            skeleton_data, labels, vid_idx, frame_pad, object_data = data
            # print(vid_idx)
            ###################################################################
            test_object_data(object_data, with_obj=args.with_obj)
            test_skeleton_data(skeleton_data)
            ###################################################################
            if args.model_args['custom_A']:
                if args.arch == 'EGCN':
                    inputs = stack_inputs_EGCN(skeleton_data, object_data)
                else:
                    inputs = stack_inputs(skeleton_data, object_data)
            else:
                inputs = process_skeleton_data(args.arch, skeleton_data, object_data, args.with_obj)
            
            ###################################################################
            # test_inputs(arch, inputs, skeleton_data, object_data, args.with_obj)
            ###################################################################

            inputs = Variable(inputs.cuda(), requires_grad=True)
            
            labels = Variable(labels.cuda())

            # Score and feature(ignored)
            # logits: [N x Classes]
            logits, features = model(inputs)
            # intensity: N x V
            intensity = get_intensity(features)
            # Length
            t = args.frames_per_clip


            # Interpolate over all frames in the clip
            # logits after interpolate: N x classes x T GPU tensor
            logits = torch.nn.functional.interpolate(logits.unsqueeze(-1), t, mode='linear', align_corners=True)
            # intensity after interpolation: N x V x T cpu tensor
            intensity = torch.nn.functional.interpolate(intensity.unsqueeze(-1), t, mode='linear', align_corners=True)

            # print("intensity size after interpo:", intensity.size())
            # print("logits interpo:", logits.size())
            # print("labels interpo:", labels.size())


            # logits: N x num_classes x T
            acc = i3d_utils.accuracy_v2(torch.argmax(logits, dim=1), torch.argmax(labels, dim=1))
            avg_acc.append(acc.item())
            logits = logits.permute(0, 2, 1)  # [ batch, frames, classes]
            labels = labels.permute(0, 2, 1)  # [ batch, frames, classes]
            intensity = intensity.permute(0, 2, 1) #[N, T, V]
            # print("intensity permute:", intensity.size())
            # print("logits permute:", logits.size())
            # print("labels permute:", labels.size())
            # logits: N*T x classes
            logits = logits.reshape(inputs.shape[0] * frames_per_clip, -1)
            labels = labels.reshape(inputs.shape[0] * frames_per_clip, -1)
            intensity = intensity.reshape(inputs.shape[0] * frames_per_clip, -1)
            # print("intensity reshape:", intensity.size())
            # print("logits reshape:", logits.size())
            # print("labels reshape:", labels.size())

            pred_labels = torch.argmax(logits, dim=1).detach().cpu().numpy().tolist()
            true_labels = torch.argmax(labels, dim=1).detach().cpu().numpy().tolist()
            intensity = intensity.numpy().tolist()
            # print("logits argmax:", np.shape(pred_labels))
            # print("labels argmax:", np.shape(true_labels))
            pred.append(pred_labels)
            true.append(true_labels)

            # print('pred_labels:',np.shape(pred_labels))
            # print('true_labels:',np.shape(true_labels))
            # frame_pad = torch.tensor([0])
            intensity_per_video = \
                utils.accume_per_video_predictions(vid_idx, frame_pad, intensity_per_video, intensity, frames_per_clip)
            pred_labels_per_video = \
                utils.accume_per_video_predictions(vid_idx, frame_pad, pred_labels_per_video, pred_labels, frames_per_clip)
            true_labels_per_video = \
                utils.accume_per_video_predictions(vid_idx, frame_pad, true_labels_per_video, true_labels, frames_per_clip)
        
    pred_labels_per_video = [np.array(pred_video_labels) for pred_video_labels in pred_labels_per_video]
    true_labels_per_video = [np.array(true_video_labels) for true_video_labels in true_labels_per_video]
    intensity_per_video = [np.array(intensity) for intensity in intensity_per_video]
    # print("len inten:", len(intensity_per_video))
    # for i in intensity_per_video:
    #     print("video len:", i.shape)

    f1_10 = metrics.f1_at_k(true_labels_per_video, pred_labels_per_video, num_classes=num_classes, overlap=0.1)
    f1_25 = metrics.f1_at_k(true_labels_per_video, pred_labels_per_video, num_classes=num_classes, overlap=0.25)
    f1_50 = metrics.f1_at_k(true_labels_per_video, pred_labels_per_video, num_classes=num_classes, overlap=0.5)
    # Compute confusion matrix
    c_matrix = confusion_matrix(np.concatenate(true).ravel(), np.concatenate(pred).ravel(),
                                labels=range(num_classes))
    class_names = utils.squeeze_class_names(test_dataset.action_list)
    # print("action list:", test_dataset.action_list)
    # print("class names list:", class_names)
    fig, ax = utils.plot_confusion_matrix(cm=c_matrix,
                      target_names=class_names,
                      title='Confusion matrix',
                      cmap=None,
                      normalize=True)

    plt.savefig(os.path.join(args.logdir, 'confusion_matrix.png'))

    accum_acc = accuracy_score(np.concatenate(true_labels_per_video).ravel(), np.concatenate(pred_labels_per_video).ravel())

    concat_acc = accuracy_score(np.concatenate(true).ravel(), np.concatenate(pred).ravel())
    print("\n"+"=========="*8 + "Test Report\n")
    print('Model:', args.arch)
    print('Model args:', args.model_args)
    print('Average accuracy(framewise):', np.mean(avg_acc))
    print('Accuracy concat video frames:', concat_acc)
    print('Accuracy accum video frames:', accum_acc)
    print("F1@10%:", f1_10)
    print("F1@25%:", f1_25)
    print("F1@50%:", f1_50)
    print("Confusion matrix: Done")

    # Store prediction and true results
    np.savez(os.path.join(args.logdir, 'prediction.npz'), *pred_labels_per_video)
    np.savez(os.path.join(args.logdir, 'true.npz'), *true_labels_per_video)
    np.savez(os.path.join(args.logdir, 'intensity.npz'), *intensity_per_video)
    # Write test report
    text_path = './log/evaluation_all_append.txt'
    with open(text_path, 'a') as f:
        f.write('---------------------------------\n')
        f.write('Model:'+ args.arch + '\n')
        f.write('Config:'+ args.config + '\n')
        f.write('With Object:'+ str(args.with_obj) + '\n')
        f.write('Acc:'+ str(round(np.mean(avg_acc), 2)) + '\n')
        f.write('Acc(concat):'+ str(round(concat_acc, 2)) + '\n')
        f.write('Acc(accum):'+ str(round(accum_acc, 2)) + '\n')
        f.write('F1@10:'+ str(round(f1_10, 2)) + '\n')
        f.write('F1@25:'+ str(round(f1_25, 2)) + '\n')
        f.write('F1@50:'+ str(round(f1_50, 2)) + '\n')
        f.write('---------------------------------\n')

        



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
    parser = init_parser()
    args = parser.parse_args()
    args = update_parameters(parser, args)
    #print(args)

    # Set GPU index
    if not args.gpu_idx == 999:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_idx)  # non-functional
        torch.cuda.set_device(0)

    run(args)
