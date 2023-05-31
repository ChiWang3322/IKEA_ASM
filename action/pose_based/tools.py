import os, logging, math, time, sys, argparse, numpy as np, copy, time, yaml, logging, datetime, shutil, math, json
from yaml.loader import SafeLoader

import torch
import pandas as pd
import agcn, st_gcn
from EfficientGCN.nets import EfficientGCN as EGCN

from matplotlib import pyplot as plt
from thop import profile

def printlog(info):
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("\n"+"=========="*8 + "%s"%nowtime)
    print(str(info)+"\n")

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
        # 1 frame velocity and 2 frames velocity
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

def test_object_data(object_data, with_obj):
    # .any() method, False If the array contains only zeros
    contains_none_zero = object_data.numpy().any()

    # Should contain non-zero element
    if with_obj:
        assert contains_none_zero == True, "With object data but object data contains only zero"
    else:
        assert contains_none_zero == False, "Without object data but object data contains none zero"

def stack_inputs(skeleton_data, obj_data):
    """
    A function stack skeleton data and object data to generate new inputs

    Inputs
    ----------
    skeleton_data : N x C_s x T x V x M tensor
        N batch size, C number of channels, T number of frames, V number of joints
    obj_data: N x C_o x T x V x M tensor
        C_o number of channels of obj_data(bbox, cat), V_o(number of objects)
    Outputs
    ----------
    inputs: N x C_o x T x (V + V_o) x M tensor
    """
    # Skeleton data -> N x C_o x T x V x M
    N, C_s, T, V, M = skeleton_data.size()
    _, C_o, _, V, _ = obj_data.size()
    # print("old skeleton_data size:", skeleton_data.size())
    # print("old obj_data size:", obj_data.size())
    temp = obj_data[:, :, :, :6, :]
    
    obj_data = temp
    # print("obj_data:", obj_data[0, :, 0, 3, 0])
    # print("new obj_data size:", obj_data.size())
    zeros = torch.zeros((N, C_o- C_s, T, V, M))
    for i in range(int((C_o - C_s - 1)/2)):
        zeros[:, i * 2 ,:, :, :] = skeleton_data[:, 0, :, :, :]
        zeros[:, i * 2 + 1 ,:, :, :] = skeleton_data[:, 1, :, :, :]
    # N x C_o x T x V x M, 4 and 7 are hands
    skeleton_data = torch.cat([skeleton_data, zeros], dim=1)
    # skeleton_data[:, 2:4, :, :, :] = skeleton_data[:, :2, :, :, :]
    skeleton_data[:, -1, :, :, :] = -1
    skeleton_data[:, -1, :, 4, :] = 10
    skeleton_data[:, -1, :, 7, :] = 10
    # print("new skeleton data:", skeleton_data[0, :, 0, 0, 0])
    inputs = torch.cat([skeleton_data, obj_data], dim=3)
    # print("New skeleton_data size:", skeleton_data.size())
    # print("cat Inputs size:", inputs.size())
    return inputs
 
def stack_inputs_EGCN(skeleton_data, object_data):
    """
    A function stack skeleton data and object data to generate new inputs for EGCN

    Inputs
    ----------
    skeleton_data : N x C_s x T x V x M tensor
        N batch size, C number of channels, T number of frames, V number of joints
    obj_data: N x C_o x T x V x M tensor
        C_o number of channels of obj_data(bbox, cat), V_o(number of objects)
    Outputs
    ----------
    inputs: N x 3 x C_o x T x (V + V_o) x M tensor
    """
    N, C, T, V, M = skeleton_data.size()
    connection = np.array([0, 0, 4, 8, 13, 0, 11, 6, 0, 0, 3, 12, 12, 12, 0, 0, 18, 22, 13, 0, 11, 20, 12, 0, 17])

    new_input = []
    # object_data = object_data.numpy()
    # For over each batch
    for i in range(N): 
        # 5 x T x V x M
        objects = object_data[i]
        # 4 x T x V x M
        joint, velocity, bone = multi_input(skeleton_data[i], connection)
        joint = torch.from_numpy(joint)
        velocity = torch.from_numpy(velocity)
        bone = torch.from_numpy(bone)

        # Skeleton data -> N x C_o x T x V x M
        C_s, T, V, M = joint.size()
        C_o, T, V, M = objects.size()
        # print("-------------------------------------")
        # print("old joint size:", joint.size())
        # print("old obj_data size:", objects.size())
        # Select 6 objects
        temp = objects[:, :, :6, :]
        # 5(bbox,) x T x 6 x M
        obj_data = temp
        if C_o > C_s:
            zeros = torch.zeros((C_o- C_s, T, V, M))
            # N x C_o x T x V x M
            joint = torch.cat([joint, zeros], dim=0)
            joint[-1, :, :, :] = -1
            joint[-1, :, 24, :] = 10
            joint[-1, :, 10, :] = 10
            
            velocity = torch.cat([velocity, zeros], dim=0)
            velocity[-1, :, :, :] = -2
            velocity[-1, :, 24, :] = 10
            velocity[-1, :, 10, :] = 10
            
            bone = torch.cat([bone, zeros], dim=0)
            bone[-1, :, :, :] = -3
            bone[-1, :, 24, :] = 10
            bone[-1, :, 10, :] = 10
        else:
            zeros = torch.zeros((C_s - C_o + 1, T, 6, M))
            # print('Obj before cat:', obj_data.size())
            obj_data = torch.cat([obj_data, zeros], dim = 0)
            obj_data[-1, :, :, :] = obj_data[2, :, :, :]
            obj_data[2, :, :, :] = obj_data[3, :, :, :]
            zeros = torch.zeros((1, T, V, M))

            joint = torch.cat([joint, zeros], dim=0)
            joint[-1, :, :, :] = -1
            joint[-1, :, 24, :] = 10
            joint[-1, :, 10, :] = 10
            
            velocity = torch.cat([velocity, zeros], dim=0)
            velocity[-1, :, :, :] = -2
            velocity[-1, :, 24, :] = 10
            velocity[-1, :, 10, :] = 10
            
            bone = torch.cat([bone, zeros], dim=0)
            bone[-1, :, :, :] = -3
            bone[-1, :, 24, :] = 10
            bone[-1, :, 10, :] = 10
            # print('Obj after cat:', obj_data.size())
        # print("obj_data:", obj_data[:, 0, 3, 0])
        # print("new obj_data size:", obj_data.size())
        # print("new joint size:", joint.size())
        # print("new v size:", velocity.size())
        # print("new b size:", bone.size())
        # print("new skeleton data:", skeleton_data[0, :, 0, 0, 0])

        joint = torch.cat([joint, obj_data], dim=2)
        velocity = torch.cat([velocity, obj_data], dim=2)
        bone = torch.cat([bone, obj_data], dim=2)
        # print("final j size:", joint.size())
        # print("new v size:", velocity.size())
        # print("new b size:", bone.size())
        # print("final j:", joint[:, 0, 3, 0])
        # print("final j:", joint[:, 0, 20, 0])
        # print("final v:", velocity[:, 0, 3, 0])
        # print("final v:", velocity[:, 0, 20, 0])
        # print("final b:", bone[:, 0, 3, 0])
        # print("final b:", bone[:, 0, 20, 0])
        # print("-------------------------------------")
        # print("New skeleton_data size:", skeleton_data.size())
        # print("cat Inputs size:", inputs.size())
        joint = joint.numpy()
        velocity = velocity.numpy()
        bone = bone.numpy()
        
        data = []
        data.append(joint)
        data.append(velocity)
        data.append(bone)

        new_input.append(data)

    # N x 3 x C_o x T x 24 x M
    inputs = torch.tensor(np.array(new_input, dtype='f'))
    # print("inputs size:", inputs.size())
    # print("Check data----------------------------------------")
    # print("final j:", inputs[0, 0, :, 0, 2, 0])
    # print("final j:", inputs[0, 0, :, 0, 21, 0])
    # print("final v:", inputs[0, 1, :, 0, 2, 0])
    # print("final v:", inputs[0, 1, :, 0, 21, 0])
    # print("final b:", inputs[0, 2, :, 0, 2, 0])
    # print("final b:", inputs[0, 2, :, 0, 21, 0])
    # print("Check data----------------------------------------")
    return inputs

def stack_inputs_v2(skeleton_data, obj_data):
    """
    A function stack skeleton data and object data to generate new inputs

    Inputs
    ----------
    skeleton_data : N x C_s x T x V x M tensor
        N batch size, C number of channels, T number of frames, V number of joints
    obj_data: N x C_o x T x V x M tensor
        C_o number of channels of obj_data(bbox, cat), V_o(number of objects)
    Outputs
    ----------
    inputs: N x C_o x T x (V + V_o) x M tensor
    """
    num_obj = 6
    # Skeleton data -> N x C_o x T x V x M
    N, C_s, T, V, M = skeleton_data.size()
    #(center, left_top, right_bottom, left_bottom, right_top)
    _, C_o, _, _, _ = obj_data.size()
    # print("old skeleton_data size:", skeleton_data.size())
    # print("old obj_data size:", obj_data.size())
    temp = obj_data[:, :, :, :num_obj, :]
    
    obj_data = temp
    # print("obj_data:", obj_data[0, :, 0, 3, 0])
    # print("new obj_data size:", obj_data.size())
    zeros = torch.zeros((N, C_o- C_s, T, V, M))
    for i in range(int((C_o - C_s - 1)/2)):
        zeros[:, i * 2 ,:, :, :] = skeleton_data[:, 0, :, :, :]
        zeros[:, i * 2 + 1 ,:, :, :] = skeleton_data[:, 1, :, :, :]
    # N x C_o x T x V x M, 4 and 7 are hands
    skeleton_data = torch.cat([skeleton_data, zeros], dim=1)
    # skeleton_data[:, 2:4, :, :, :] = skeleton_data[:, :2, :, :, :]
    skeleton_data[:, -1, :, :, :] = -1
    skeleton_data[:, -1, :, 4, :] = 10
    skeleton_data[:, -1, :, 7, :] = 10
    
    # print("new skeleton data:", skeleton_data[0, :, 0, 0, 0])
    inputs = torch.cat([skeleton_data, obj_data], dim=3)
    N, C, T, V, M = inputs.size()
    # distance = zeros = torch.zeros((N, C_o, T, V, M))
    zeros = torch.zeros((N, num_obj, T, V, M))
    inputs = torch.cat([inputs, zeros], dim=1)
    for n in range(N):
        for t in range(T):
            for v in [4, 7, 18, 19, 20, 21, 22, 23]:
                hand = inputs[n, :, t, v, 0]
                # print("Current hand:", hand)
                for o in range(18, 18 + num_obj):
                    obj = inputs[n, :, t, o, 0]
                    # print("Current obj:", obj)
                    # print("index:", 18 + 3 - o)
                    distance = (obj[0] - hand[0])**2 + (obj[1] - hand[1])**2
                    inputs[n, o - 18 + 3, t, v, 0] = distance**0.5


    # print("Check hand1:", inputs[0, :, 0, 4, 0])
    # print("Check hand2:", inputs[0, :, 0, 7, 0])
    # print("Check object:", inputs[4, :, 20, 20, 0])
    # print("Inputs size:", inputs.size())
    # print("New skeleton_data size:", skeleton_data.size())
    # print("cat Inputs size:", inputs.size())
    return inputs

def get_intensity(features):
    """
    A function convert features in the last layer to intensity

    Inputs
    ----------
    features : N x C x T x V x M tensor
        N batch size, C number of channels, T number of frames, V number of joints
    Outputs
    intensity
    ----------
    """
    features = features.detach().cpu()
    # print(features.size())
    N, C, T, V, M = features.size()
    # N, 1, 1, V, M, summation along Channel and Time dimension
    intensity = (features*features).sum(dim=1).sum(dim=1)**0.5
    intensity = torch.squeeze(intensity, dim=-1)
    # intensity = intensity.numpy().tolist()
    # print("intensity size:",np.shape(intensity))
    # print("intensity:", intensity[0])
    return intensity

def get_model_A(logdir, model):
    count = 0
    for name, param in model.named_parameters():
        names_split = name.split('.')
        for n in names_split:
            if n == 'A':
                print(name, param.size())
                param = param.detach()
                param = param.sum(dim=0)
                plt.clf()
                plt.imshow(param.numpy(), cmap='viridis', vmin=0.0, vmax=1.0)
                plt.colorbar()
                # plt.show()
                plt.savefig(logdir + '/A_l' + str(count))
                count += 1
                print(param)
            # else:
            #     print(name, param.size())
                # plt.savefig('my_plot.png')
                # input("Press Enter to continue...")

def get_model(model_type, model_args, num_classes=33):
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
def get_model_params_flops(config):

    config_path = './configs_v1/'
    if os.path.exists(config_path + config + '.yaml'):
        with open(config_path + config + '.yaml', 'r') as f:
            try:
                yaml_arg = yaml.safe_load(f, Loader=SafeLoader)
            except:
                yaml_arg = yaml.load(f, Loader=SafeLoader)
            model = get_model(yaml_arg['arch'], yaml_arg['model_args'])

            logdir = yaml_arg['logdir']
            model_path = os.path.join(logdir, 'best_classifier.pth')
            checkpoints = torch.load(model_path)
            model.load_state_dict(checkpoints["model_state_dict"]) # load trained model
            model.cuda()

            arch = yaml_arg['arch']
            if arch == 'EGCN':
                input_tmp = torch.randn((1, 3, 5, 50, 24, 1))
            elif arch == 'AGCN' or 'STGCN':
                in_channels = yaml_arg['model_args']['in_channels']
                input_tmp = torch.randn((1, 3, 50, 24, 1))
            input_tmp = Variable(input_tmp.cuda(), requires_grad=False)
            flops, params = profile(model, inputs=(input_tmp,))
            print('FLOPs = ' + str(flops/1000**3) + 'G')
    else:
        raise ValueError('Do NOT exist this config file: {}'.format(config))
def reorder_skeleton(pose_json_filename):
    with open(pose_json_filename) as json_file:
        data = json.load(json_file)
        data = data[0]
        keys = list(data.keys())
        print(keys)
    pose = []
    for i in range(len(keys)):
        key = keys[i]
        joint = [data[key]['x'], data[key]['y']]
        pose.append(joint)
    # print(pose)
                
    # print(data.keys())
def get_obj_data(scan_path):
    obj_json_filename = os.path.join(scan_path, '2d_objects', 'frame_0.json')
    with open(obj_json_filename) as json_file:
        data = json.load(json_file)
    csv_file = os.path.join(scan_path, 'rgb','metadata.csv')
    metadata = pd.read_csv(csv_file)
    metadata = metadata.set_index('name')
    Height = float(metadata.loc['frameHeight',:]['value'])
    Width = float(metadata.loc['frameWidth',:]['value'])
    # print("Width:{}, Height:{}".format(Width, Height))
    obj_data = []
    for obj in data:
        bbox = obj['bounding_box']
        h, w, x, y = bbox['h'] * Height, bbox['w'] * Width, bbox['x'] * Width, bbox['y'] * Height
        obj_data.append((h, w, x, y))
    # print(obj_data)
if __name__=='__main__':
    # logdir = './log/AGCN_50_w_obj_A'
    # num_classes=33
    # # model = st_gcn.Model(num_class=num_classes, in_channels=11
    # #                     ,graph_args={'layout': 'openpose', 'strategy': 'spatial'}
    # #                     ,edge_importance_weighting= True
    # #                     ,dropout= 0.2
    # #                     ,custom_A= True)
    # model =  agcn.Model(num_class=num_classes, num_point=24, num_person=1
    #                     ,graph='graph.kinetics.Graph'
    #                     ,graph_args={'labeling_mode':'spatial'}
    #                     ,in_channels=11
    #                     ,custom_A=True)
    # model_path = os.path.join(logdir, 'best_classifier.pth')
    # checkpoints = torch.load(model_path)
    # model.load_state_dict(checkpoints["model_state_dict"]) # load trained model
    # get_model_A(logdir, model)
    pose_path = '/media/zhihao/Chi_SamSungT7/KIT_Bimanual/images/subject_1/task_2_k_cooking_with_bowls/take_1/body_pose'
    scan_path = '/media/zhihao/Chi_SamSungT7/KIT_Bimanual/images/subject_1/task_2_k_cooking_with_bowls/take_1'
    # reorder_skeleton(pose_path)
    get_obj_data(scan_path)