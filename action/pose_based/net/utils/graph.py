import numpy as np
from matplotlib import pyplot as plt
class Graph():
    """ The Graph to model the skeletons extracted by the openpose

    Args:
        strategy (string): must be one of the follow candidates
        - uniform: Uniform Labeling
        - distance: Distance Partitioning
        - spatial: Spatial Configuration
        For more information, please refer to the section 'Partition Strategies'
            in our paper (https://arxiv.org/abs/1801.07455).

        layout (string): must be one of the follow candidates
        - openpose: Is consists of 18 joints. For more information, please
            refer to https://github.com/CMU-Perceptual-Computing-Lab/openpose#output
        - ntu-rgb+d: Is consists of 25 joints. For more information, please
            refer to https://github.com/shahroudy/NTURGB-D

        max_hop (int): the maximal distance between two connected nodes
        dilation (int): controls the spacing between the kernel points

    """

    def __init__(self,
                 layout='openpose',
                 strategy='uniform',
                 max_hop=1,
                 dilation=1):
        self.max_hop = max_hop
        self.dilation = dilation

        self.get_edge(layout)
        self.hop_dis = get_hop_distance(self.num_node, self.edge, max_hop=max_hop)
        self.get_adjacency(strategy)

    def __str__(self):
        return self.A

    def get_edge(self, layout):
        if layout == 'openpose':
            self.num_node = 18
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [(4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12,
                                                                        11),
                             (10, 9), (9, 8), (11, 5), (8, 2), (5, 1), (2, 1),
                             (0, 1), (15, 0), (14, 0), (17, 15), (16, 14)]
            self.edge = self_link + neighbor_link
            self.center = 1
        elif layout == 'ntu-rgb+d':
            self.num_node = 25
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21),
                              (6, 5), (7, 6), (8, 7), (9, 21), (10, 9),
                              (11, 10), (12, 11), (13, 1), (14, 13), (15, 14),
                              (16, 15), (17, 1), (18, 17), (19, 18), (20, 19),
                              (22, 23), (23, 8), (24, 25), (25, 12)]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            self.edge = self_link + neighbor_link
            self.center = 21 - 1
        elif layout == 'ntu_edge':
            self.num_node = 24
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [(1, 2), (3, 2), (4, 3), (5, 2), (6, 5), (7, 6),
                              (8, 7), (9, 2), (10, 9), (11, 10), (12, 11),
                              (13, 1), (14, 13), (15, 14), (16, 15), (17, 1),
                              (18, 17), (19, 18), (20, 19), (21, 22), (22, 8),
                              (23, 24), (24, 12)]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            self.edge = self_link + neighbor_link
            self.center = 2
        # elif layout=='customer settings':
        #     pass
        elif layout == 'openpose_object_hand':
            # Object connects only to two hand joints
            self.num_node = 24
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [(4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12, 11),
                             (10, 9), (9, 8), (11, 5), (8, 2), (5, 1), (2, 1),
                             (0, 1), (15, 0), (14, 0), (17, 15), (16, 14),
                            (18, 4), (19, 4), (20, 4), (21, 4), (22, 4), (23, 4),
                            (18, 7), (19, 7), (20, 7), (21, 7), (22, 7), (23, 7), ]
            self.edge = self_link + neighbor_link
            self.center = 1
        ################################################################################
        elif layout == 'KIT_object_object_hand':
            # Join 25 points, objects 6
            self.num_node = 31
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [# Joint connection
                             (16, 18), (18, 4), (4, 2), (18, 13), (4, 13), (13, 12),
                             (22, 12), (22, 17), (17, 24),
                             (12, 8), (8, 3), (3, 10),
                             (12, 11), (20, 11), (11, 6),
                             (20, 21), (21, 10),
                             (6, 7), (7, 0),
                             # Hand object connection
                             (25, 24), (26, 24), (27, 24), (28, 24), (29, 24), (30, 24),
                             (25, 10), (26, 10), (27, 10), (28, 10), (29, 10), (30, 10)
                            ]
            self.edge = self_link + neighbor_link
            self.center = 1
        ################################################################################
        elif layout == 'openpose_object_object_hand':
            self.num_node = 24
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [(4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12, 11),
                            (10, 9), (9, 8), (11, 5), (8, 2), (5, 1), (2, 1),
                            (0, 1), (15, 0), (14, 0), (17, 15), (16, 14),
                            (18, 4), (19, 4), (20, 4), (21, 4), (22, 4), (23, 4),
                            (18, 7), (19, 7), (20, 7), (21, 7), (22, 7), (23, 7), 
                            (18, 19), (18, 20), (18, 21), (18, 22), (18, 23),
                            (19, 20), (19, 21), (19, 22), (19, 23),
                            (20, 21), (20, 22), (20, 23),
                            (21, 22), (21, 23),
                            (22, 23)]
            self.edge = self_link + neighbor_link
            self.center = 1
        elif layout == 'openpose_object_joint':
            self.num_node = 24
            self_link = [(i, i) for i in range(self.num_node)]
            # joint connection + object and hand connection
            neighbor_link = [(4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12, 11),
                            (10, 9), (9, 8), (11, 5), (8, 2), (5, 1), (2, 1),
                            (0, 1), (15, 0), (14, 0), (17, 15), (16, 14),
                            (18, 4), (19, 4), (20, 4), (21, 4), (22, 4), (23, 4),
                            (18, 7), (19, 7), (20, 7), (21, 7), (22, 7), (23, 7)]
            # Append object-joint connection
            for obj_num in [18, 19, 20, 21, 22, 23]:
                for joint_num in range(18):
                    neighbor_link.append((obj_num, joint_num))
            self.edge = self_link + neighbor_link
            self.center = 1
        else:
            raise ValueError("Do Not Exist This Layout.")

    #计算邻接矩阵A
    def get_adjacency(self, strategy):
        valid_hop = range(0, self.max_hop + 1, self.dilation)  #range(start,stop,step)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = normalize_digraph(adjacency)

        if strategy == 'uniform':
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = normalize_adjacency
            self.A = A
        elif strategy == 'distance':
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis ==
                                                                hop]
            self.A = A
        elif strategy == 'spatial':
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            if self.hop_dis[j, self.center] == self.hop_dis[
                                    i, self.center]:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif self.hop_dis[j, self.
                                              center] > self.hop_dis[i, self.
                                                                     center]:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            A = np.stack(A)
            self.A = A
        else:
            raise ValueError("Do Not Exist This Strategy")

# 此函数的返回值hop_dis就是图的邻接矩阵
def get_hop_distance(num_node, edge, max_hop=1):
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1

    # compute hop steps
    hop_dis = np.zeros((num_node, num_node)) + np.inf  # np.inf 表示一个无穷大的正数
    # np.linalg.matrix_power(A, d)求矩阵A的d幂次方,transfer_mat矩阵(I,A)是一个将A矩阵拼接max_hop+1次的矩阵
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    # (np.stack(transfer_mat) > 0)矩阵中大于0的返回Ture,小于0的返回False,最终arrive_mat是一个布尔矩阵,大小与transfer_mat一样
    arrive_mat = (np.stack(transfer_mat) > 0)
    # range(start,stop,step) step=-1表示倒着取
    for d in range(max_hop, -1, -1):
        # 将arrive_mat[d]矩阵中为True的对应于hop_dis[]位置的数设置为d
        hop_dis[arrive_mat[d]] = d
    return hop_dis

# 将矩阵A中的每一列的各个元素分别除以此列元素的形成新的矩阵
def normalize_digraph(A):
    Dl = np.sum(A, 0) #将矩阵A压缩成一行
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)
    AD = np.dot(A, Dn)
    return AD


def normalize_undigraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-0.5)
    DAD = np.dot(np.dot(Dn, A), Dn)
    return DAD

if __name__ == '__main__':
    g = Graph(layout='openpose_object_object_hand')
    print(g.A)
    print(g.A.shape)
    plt.imshow(g.A[0])
    plt.show()
