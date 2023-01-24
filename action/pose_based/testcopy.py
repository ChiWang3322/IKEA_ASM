import numpy as np
from pyrutils import metrics
from net.utils.graph import Graph
import matplotlib.pyplot as plt
# with open('test_true.npy', 'rb') as f:
#         true = np.load(f, allow_pickle=True)

# with open('test_pred.npy', 'rb') as f:
#         pred = np.load(f, allow_pickle=True)

# f1_10_v2 = metrics.f1_at_k_v2(true, pred, num_classes=33, overlap=0.10)
# f1_25_v2 = metrics.f1_at_k_v2(true, pred, num_classes=33, overlap=0.25)
# f1_50_v2 = metrics.f1_at_k_v2(true, pred, num_classes=33, overlap=0.50)
# f1_10 = metrics.f1_at_k(true, pred, num_classes=33, overlap=0.10)
# f1_25 = metrics.f1_at_k(true, pred, num_classes=33, overlap=0.25)
# f1_50 = metrics.f1_at_k(true, pred, num_classes=33, overlap=0.50)
# print('-------------------')
# print('f1@10_v1:', f1_10)
# print('f1@25_v1:', f1_25)
# print('f1@50_v1:', f1_50)
# print('-------------------')
# print('f1@10_v2:', f1_10_v2)
# print('f1@25_v2:', f1_25_v2-0.000112378124129)
# print('f1@50_v2:', f1_50_v2)

# graph_args = {'layout': 'openpose', 'strategy': 'spatial'}
# graph_A = Graph(**graph_args)
# print(np.shape(graph_A.A))
