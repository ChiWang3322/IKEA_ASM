import numpy as np
import sys
from matplotlib import pyplot as plt
sys.path.extend(['../'])
from graph import tools

# import networkx as nx

# Joint index:
# {0,  "Nose"}
# {1,  "Neck"},
# {2,  "RShoulder"},
# {3,  "RElbow"},
# {4,  "RWrist"},
# {5,  "LShoulder"},
# {6,  "LElbow"},
# {7,  "LWrist"},
# {8,  "RHip"},
# {9,  "RKnee"},
# {10, "RAnkle"},
# {11, "LHip"},
# {12, "LKnee"},
# {13, "LAnkle"},
# {14, "REye"},
# {15, "LEye"},
# {16, "REar"},
# {17, "LEar"},

# Edge format: (origin, neighbor)

self_link_18 = [(i, i) for i in range(18)]
inward_18 = [(4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12, 11), (10, 9), (9, 8),
          (11, 5), (8, 2), (5, 1), (2, 1), (0, 1), (15, 0), (14, 0), (17, 15),
          (16, 14)]
outward_18 = [(j, i) for (i, j) in inward_18]
neighbor_18 = inward_18 + outward_18

self_link_24 = [(i, i) for i in range(24)]
inward_24 = [(4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12, 11), (10, 9), (9, 8),
          (11, 5), (8, 2), (5, 1), (2, 1), (0, 1), (15, 0), (14, 0), (17, 15),(16, 14), 
          (18, 4), (19, 4), (20, 4), (21, 4), (22, 4), (23, 4),
          (18, 7), (19, 7), (20, 7), (21, 7), (22, 7), (23, 7), 
        #   (18, 19), (18, 20), (18, 21), (18, 22), (18, 23),
        #   (19, 20), (19, 21), (19, 22), (19, 23),
        #   (20, 21), (20, 22), (20, 23),
        #   (21, 22), (21, 23),
        #   (22, 23)
          ]
outward_24 = [(j, i) for (i, j) in inward_24]
neighbor_24 = inward_24 + outward_24


class Graph:
    def __init__(self, labeling_mode='spatial', num_node=18):
        
        self.num_node = num_node
        if self.num_node == 18:
            self.self_link = self_link_18
            self.inward = inward_18
            self.outward = outward_18
            self.neighbor = neighbor_18
        else:
            self.self_link = self_link_24
            self.inward = inward_24
            self.outward = outward_24
            self.neighbor = neighbor_24
        self.A = self.get_adjacency_matrix(labeling_mode)
        # print("shape of A:", np.shape(self.A))
        print("A pattern: self link, inward, outward")
    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(self.num_node, self.self_link, self.inward, self.outward)
        else:
            raise ValueError()
        return A


if __name__ == '__main__':
    g = Graph('spatial', 24)
    A = g.A
    # print(A.shape)
    A = A.sum(axis=0)
    print(A)
    plt.imshow(A, cmap='viridis', vmin=0.0, vmax=1.0)
    plt.colorbar()
    plt.show()
