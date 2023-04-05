import os, logging, math, time, sys, argparse, numpy as np, copy, time, yaml, logging, datetime, shutil, math
from yaml.loader import SafeLoader
import torch

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

def test_object_data(object_data, with_obj=args.with_obj):
    # .any() method, False If the array contains only zeros
    contains_none_zero = object_data.numpy().any()

    # Should contain non-zero element
    if with_obj:
        assert contains_none_zero == True, "With object data but object data contains only zero"
    else:
        assert contains_none_zero == False, "Without object data but object data contains none zero"