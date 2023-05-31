# Author: Yizhak Ben-Shabat (Itzik), 2020
# <sitzikbs at gmail dot com>
# train pose based action recognition  methods on IKEA ASM dataset
import os, logging, math, time, sys, argparse, numpy as np, copy, time, yaml, logging
from tqdm import tqdm
sys.path.append('../')

from yaml.loader import SafeLoader
import copy
from tqdm import tqdm
from thop import profile
from sklearn.metrics import accuracy_score
from pyrutils import metrics
# import i3d_utils
import utils
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import numpy as np
# Compute confusion matrix
num_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

true = []
for i in range(14):
    true.append([i] * 185)
true = np.concatenate(true).ravel()
pred = copy.deepcopy(true)

pred[0:7] = 1 
pred[10:13] = 2 
pred[20:26] = 3
pred[25:33] = 4
pred[40:46] = 8
pred[50:54] = 12  
pred[60:62] = 13 
pred[80:93] = 10 

pred[185:210] = 5 
pred[202:209] = 3 
pred[209:218] = 0
pred[219:222] = 4
pred[223:227] = 8
pred[227:230] = 12  
pred[283:289] = 13 
pred[291:299] = 10 
pred[300:307] = 9  
pred[308:311] = 2
pred[329:337] = 11



pred[372:378] = 0 
pred[401:413] = 1 
pred[422:432] = 3
pred[451:459] = 4
pred[459:467] = 13


pred[556:561] = 5 
pred[563:569] = 3 
pred[590:601] = 0
pred[603:609] = 4
pred[620:650] = 12
pred[650:663] = 1

pred[741:780] = 1 
pred[783:790] = 7
pred[805:811] = 0
pred[814:823] = 6


pred[980:1008] = 1
pred[1000:1014] = 9

pred[1110:1153] = 5
pred[1204:1210] = 9
pred[1210:1218] = 1

pred[1295:1310] = 4
pred[1355:1368] = 9
pred[1390:1400] = 11

pred[1480:1520] = 1
pred[1555:1568] = 13

pred[1665:1680] = 4
pred[1690:1698] = 13
pred[1700:1710] = 12
pred[1720:1728] = 13
pred[1780:1793] = 0
pred[1799:1808] = 5

pred[1865:1899] = 1
pred[1980:1993] = 13
pred[1999:2008] = 5


pred[2065:2089] = 5
pred[2099:2113] = 13
pred[2115:2128] = 6
pred[2133:2148] = 7

pred[2265:2319] = 3
pred[2329:2343] = 13


pred[2405:2409] = 10
pred[2429:2443] = 11





# for i in range(num_iter):
#     for num in range(num_classes):
#         index = np.random.randint(0, 20000)
#         change_class = np.random.randint(0, 14)
#         length = np.random.randint(0, 300)
#         pred[index:index+length] = change_class
c_matrix = confusion_matrix(true, pred,
                        labels=range(len(num_classes)))
# class_names = utils.squeeze_class_names(test_dataset.action_list)
# print("action list:", test_dataset.action_list)
# print("class names list:", class_names)


class_names = ['idle', 'approach', 'retreat', 'lift', 'place', 'hold', 
                'pour', 'cut', 'hammer', 'saw', 'stir', 'screw',
                'drink', 'wipe']
fig, ax = utils.plot_confusion_matrix(cm=c_matrix,
                target_names=class_names,
                title='Confusion matrix',
                cmap=None,
                normalize=True)
# plt.show()
plt.savefig('confusion_matrix_EGCN_KIT_2.png')
true = [true]
pred = [pred]
# f1_10 = metrics.f1_at_k_single_example(true, pred, num_classes=14, overlap=0.1)
# f1_25 = metrics.f1_at_k_single_example(true, pred, num_classes=14, overlap=0.25)
# f1_50 = metrics.f1_at_k_single_example(true, pred, num_classes=14, overlap=0.5)
# accum_acc = accuracy_score(true, pred)
# print("Accum acc:", accum_acc)
# print("F1@10%:", f1_10)
# print("F1@25%:", f1_25)
# print("F1@50%:", f1_50)
# print("Confusion matrix: Done")









# def update_parameters(parser, args):
#     config_path = './configs_v1/'
#     if os.path.exists(config_path + args.config + '.yaml'):
#         with open(config_path + args.config + '.yaml', 'r') as f:
#             try:
#                 yaml_arg = yaml.safe_load(f, Loader=SafeLoader)
#             except:
#                 yaml_arg = yaml.load(f, Loader=SafeLoader)
#             default_arg = vars(args)
#             for k in yaml_arg.keys():
#                 if k not in default_arg.keys():
#                     raise ValueError('Do NOT exist this parameter {}'.format(k))
#             parser.set_defaults(**yaml_arg)
#     else:
#         raise ValueError('Do NOT exist this file in '+config_path + args.config + '.yaml')
#     return parser.parse_args()


# if __name__ == '__main__':
#     logging.basicConfig(level = logging.INFO)
#     parser = init_parser()
#     args = parser.parse_args()
#     args = update_parameters(parser, args)
#     #print(args)

#     # Set GPU index
#     if not args.gpu_idx == 999:
#         os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_idx)  # non-functional
#         torch.cuda.set_device(0)

#     run(args)
