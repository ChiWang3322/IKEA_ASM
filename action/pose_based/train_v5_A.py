# Author: Yizhak Ben-Shabat (Itzik), 2020
# <sitzikbs at gmail dot com>
# train pose based action recognition  methods on IKEA ASM dataset


import os, logging, math, time, sys, argparse, numpy as np, copy, time, yaml, logging, datetime, shutil
from yaml.loader import SafeLoader
from tqdm import tqdm
from thop import profile

sys.path.append('../') # pose_based
sys.path.append('../clip_based/i3d/')  # for utils and video transforms
sys.path.append('../..')    #for data loader
sys.path.append('../../..')# for dataset
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchviz import make_dot
import torch.optim.lr_scheduler as lr_scheduler

import st_gcn, agcn, st2ransformer_dsta
from EfficientGCN.nets import EfficientGCN as EGCN

from Obj_stream import ObjectNet

import i3d_utils as utils


from EfficientGCN.activations import *
from IKEAActionDataset import IKEAPoseActionVideoClipDataset as Dataset
from torch.utils.tensorboard import SummaryWriter
from tools import *



class CosineWarmupLR(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, max_epochs, warmup_factor=0.1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_factor = warmup_factor
        self.cycle_epochs = max_epochs - warmup_epochs
        super(CosineWarmupLR, self).__init__(optimizer)
        
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            return [base_lr * self.warmup_factor * ((self.last_epoch+1)/self.warmup_epochs) for base_lr in self.base_lrs]
        else:
            progress = (self.last_epoch - self.warmup_epochs) / self.cycle_epochs
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return [base_lr * cosine_decay for base_lr in self.base_lrs] 

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


#########################################
######## Modify adjacency matrix ########
#########################################
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
    
    elif model_type == 'DSTA':

        model = st2ransformer_dsta.DSTANet(num_class=num_classes,**model_args)  
    else:
        raise ValueError("Unsupported architecture: please select EGCN | STGCN | AGCN | STGAN")
    return model

# def import_class(name):
#     components = name.split('.')
#     mod = __import__(components[0])
#     for comp in components[1:]:
#         mod = getattr(mod, comp)
#     return mod

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
    train_dataset = Dataset(args.dataset_path, db_filename=args.db_filename, train_filename="train_cross_env_small.txt",
                        test_filename=args.test_filename,
                        transform=train_transforms, set='train', camera=args.camera, frame_skip=args.frame_skip,
                        frames_per_clip=args.frames_per_clip, mode=args.load_mode, pose_path=args.pose_relative_path, 
                        arch=args.arch, obj_path=args.obj_path, with_obj=args.with_obj)
    # set the split ratio
    split_ratio = 0.8

    # split the dataset into training and validation sets
    np.random.seed(42)
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
                                                  pin_memory=True, sampler=train_sampler, shuffle=False)
    test_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, sampler=val_sampler, 
                                                    num_workers=8,pin_memory=True, shuffle=False)


    
    # setup the model
    arch = args.arch
    num_classes = train_dataset.num_classes

    model_args = args.model_args

    print("\n"+"=========="*8 + "Dataset INFO\n")
    print('Dataset: IKEA_ASM')
    print('Dataset path:', args.dataset_path)
    print('Number of clips in the whole dataset(train_cross_env.txt):', int(len(train_dataset)))
    print("Number of clips in the train dataset:{}".format(int(len(train_dataset)*0.8)))
    print("Number of clips in the test dataset:{}".format(int(len(train_dataset)*0.2)))
    print("Classes:{}\nbatch_size:{}\nframe_skip:{}\nframes_per_clip:{}".format(num_classes, args.batch_size, args.frame_skip, args.frames_per_clip))
    print("Device:{}\ntrain filename:{}\ntest_filename:{}".format(args.camera, args.train_filename, args.test_filename))
    print("Object Included:{}".format(args.with_obj))

    
    def print_model_parameters(model):
        print("Trainable parameters...............")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name, param.shape)


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
    # Here use import class easier
    model = init_model(args.arch, args.model_args, num_classes)
    # print_model_parameters(model)
    ########################################
    ##### Insert code here for refine ######
    ########################################

    if not next(model.parameters()).is_cuda:
        model.cuda()

    # print_model_parameters(model)
    # Init optimizer
    optim_args = args.optimizer_args
    optim_type = args.optimizer
    opt_args = optim_args[optim_type]
    if optim_type == 'Adam':
        optimizer = optim.Adam(model.parameters(), **opt_args)
    elif optim_type == 'SGD':
        optimizer = optim.SGD(model.parameters(), **opt_args)
    # Init learning rate scheduler

    warmup_epochs = 15
    warmup_factor = 0.4
    lr_sched = CosineWarmupLR(optimizer, warmup_epochs=warmup_epochs, max_epochs=args.n_epochs, warmup_factor=warmup_factor)


    # lr_sched = optim.lr_scheduler.MultiStepLR(optimizer, **(args.scheduler_args['MultiStepLR']))


    # Print Optimizer INFO
    print("\n"+"=========="*8 + "Optimizer and SCheduler INFO\n")
    print('Using optimizer:{}'.format(optim_type))
    print('Optimizer args:', opt_args)
    print('Scheduler:{}, warmp-up epochs: {}, warm-up factor:{}'.format('CosineWarmupLR', 
                                                        warmup_epochs, 
                                                        warmup_factor))

    model_total_params = sum(p.numel() for p in model.parameters())
    model_total_params = model_total_params / 10**6

    # Print model INFO
    print("\n"+"=========="*8 + "Model INFO\n")
    print('Model:{}\nnumber of parameters:{}M'.format(arch, round(model_total_params, 2)))
    print('Model args:', model_args)
    print('Graph layout:', model_args['layout'])
    print('Logdir:', args.logdir)
    # input_tmp = torch.randn((1, 3, 5, 50, 24, 1))
    # input_tmp = Variable(input_tmp.cuda(), requires_grad=False)

    # flops, params = profile(model, inputs=(input_tmp,))
    # print('FLOPs = ' + str(flops/1000**3) + 'G')

    # train iterations
    train_num_batch = len(train_dataloader)
    test_num_batch = len(test_dataloader)


    best_acc = 0


    max_steps = args.n_epochs


    #for epoch in range(num_epochs):
    for steps in range(max_steps):
        # Log info
        curr_lr = lr_sched.get_last_lr()
        printlog("Epoch {0} / {1}, learning rate: {2}".format(steps, max_steps, curr_lr))
        writer.add_scalar("Current lr", torch.tensor(curr_lr), steps)
        # Initialization
        train_loss = 0.0
        num_iter = 0
        optimizer.zero_grad()
        
        # Iterate over data.
        train_acc = []
        loop = tqdm(enumerate(train_dataloader), total =len(train_dataloader), file = sys.stdout)
        # Training phase
        for train_batchind, data in loop:
            model.train(True)
            num_iter += 1
            # get the inputs
            skeleton_data, labels, vid_idx, frame_pad, object_data = data
            
            ###################################################################
            test_object_data(object_data, with_obj=args.with_obj)
            test_skeleton_data(skeleton_data)
            ###################################################################
            if args.arch == 'EGCN':
                inputs = stack_inputs_EGCN(skeleton_data, object_data)
                # print("Inputs size:", inputs.size())
                # print("Check hand1 joint:", inputs[0, 0, :, 20, 4, 0])
                # print("Check hand2 joint:", inputs[0, 0, :, 20, 4, 0])
                # print("Check object:", inputs[0, 0, :, 20, 20, 0])
            else:
                # start = time.time()
                inputs = stack_inputs(skeleton_data, object_data)
                # print("process time:", time.time()-start)
            # print("Inputs size:", inputs.size())
            # print("Check hand1:", inputs[0, :, 0, 4, 0])
            # print("Check hand2:", inputs[0, :, 0, 7, 0])
            # print("Check object:", inputs[4, :, 20, 20, 0])
            
            ###################################################################
            # test_inputs(arch, inputs, skeleton_data, object_data, args.with_obj)
            ###################################################################

            inputs = Variable(inputs.cuda(), requires_grad=True)
            # input_tmp = inputs[0:2]
            # print("size:", input_tmp.size())
            # flops, params = profile(model, inputs=(input_tmp,))
            # print('FLOPs = ' + str(flops/2/1000**3) + 'G')
            # print('Params = ' + str(params/1000**2) + 'M')
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
            # make_dot(loss).render("gradient_flow")
            # Calculate accuracy
            acc = utils.accuracy_v2(torch.argmax(per_frame_logits, dim=1), labels)

            
            train_acc.append(acc.item())
            loop.set_postfix({'train_loss': round(train_loss/(train_batchind+1), 3), 'train_acc': round(np.mean(train_acc), 3)})
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
            loop = tqdm(enumerate(test_dataloader), total =len(test_dataloader), file = sys.stdout)
            for test_batchind, data in loop:
                skeleton_data, labels, vid_idx, frame_pad, object_data = data
                if args.arch == 'EGCN':
                    inputs = stack_inputs_EGCN(skeleton_data, object_data)
                else:
                    
                    inputs = stack_inputs(skeleton_data, object_data)

                # test_inputs(arch, inputs, skeleton_data, object_data, args.with_obj)

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
                loop.set_postfix({'val_loss': round(val_loss/(test_batchind+1), 3), 'val_acc': round(np.mean(val_acc), 3)})

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
                        "optimizer_state_dict": optimizer.state_dict()}, os.path.join(args.logdir, 'best_classifier.pth'))
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
