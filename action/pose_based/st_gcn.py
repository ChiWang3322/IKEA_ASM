import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from net.utils.graph import Graph

class Model(nn.Module):
    r"""Spatial temporal graph convolutional networks.

    Args:
        in_channels (int): Number of channels in the input data
        num_class (int): Number of classes for the classification task
        graph_args (dict): The arguments for building the graph
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph
        **kwargs (optional): Other parameters for graph convolution units

    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
        - Output: :math:`(N, num_class)` where
            :math:`N` is the batch size,
            :math:`T_{in}` is the temporal length of input sequence,
            :math:`V_{in}` is the number of graph nodes,
            :math:`M_{in}` is the number of instance(person) in a frame.
    """
    def __init__(self, in_channels, num_class, graph_args, edge_importance_weighting, custom_A=False, **kwargs):
        super().__init__()

        # Load predefined graph/adjacency matrix

        self.graph = Graph(**graph_args)
        # Predefined adjacency matrix (3 x V x V, self, inwards, outwars)
        if custom_A:
            A = self.graph.A
            shape = np.shape(A)
            # print("Old A shape:", shape)
            tmp = np.zeros((shape[0], shape[1]+6, shape[2] + 6))
            tmp[:, :shape[1], :shape[2]] = A
            A = tmp
            A = torch.tensor(A, dtype=torch.float32, requires_grad=False)   
            self.register_buffer('A', A) 
            # A = nn.Parameter(torch.from_numpy(A.astype(np.float32)), requires_grad=True)
            # self.register_parameter("A", A)


            # print("New A size:", A.size())
            # print("A:", A[0])
        else:
            A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)   
        
            self.register_buffer('A', A) 
        
        


        # Build networks
        spatial_kernel_size = A.size(0) # 3
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        kwargs0 = {k:v for k, v in kwargs.items() if k != 'dropout'}
        self.st_gcn_networks = nn.ModuleList((
            st_gcn(in_channels, 64, kernel_size, 1, residual = False, **kwargs0),
            st_gcn(64, 64, kernel_size, 1, **kwargs0),
            st_gcn(64, 64, kernel_size, 1, **kwargs0),
            st_gcn(64, 64, kernel_size, 1, **kwargs0),
            st_gcn(64, 128, kernel_size, 2, **kwargs0),
            st_gcn(128, 128, kernel_size, 1, **kwargs0),
            st_gcn(128, 128, kernel_size, 1, **kwargs0),
            st_gcn(128, 256, kernel_size, 2, **kwargs0),
            st_gcn(256, 256, kernel_size, 1, **kwargs0),
            st_gcn(256, 256, kernel_size, 1, **kwargs0),
        ))

        # Initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.st_gcn_networks   # Each layer has an edge importance matrix, corresponding to M in the paper
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)  # 1 for each layer
        
        # fcn for final prediction
        self.fcn = nn.Conv2d(256, num_class, kernel_size=1)

    def forward(self, x):

        # Data normalization
        N, C, T, V, M = x.size()    # (64, 2, 100, 18, 1)
        # Permute dimension
        x = x.permute(0, 4, 3, 1, 2).contiguous()   # (64, 1, 18, 2, 100)
        x = x.view(N * M, V * C, T) # (64, 18*3, 100)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)   # (64, 1, 18, 2, 100)
        x = x.permute(0, 1, 3, 4, 2).contiguous()   # (64, 1, 2, 100, 18)-> N, M, C, T, V
        x = x.view(N * M, C, T, V) #(N, C, T, V)

        # forward
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)
        # (64, 256, 1, 1)
        _, c, t, v = x.size()
        feature = x.view(N, M, c, t, v).permute(0, 2, 3, 4, 1)
        x = F.avg_pool2d(x, x.size()[2:])   # size of pooling layer -> (100, 18)
        x = self.fcn(x)
        x = x.view(x.size(0), -1)
        
        return x, feature
    def extract_feature(self, x):

        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # forward
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance) #（64,256,300,18）

        _, c, t, v = x.size() #（64,256,300,18）
        # feature的维度是（64,256,300,18,1）
        feature = x.view(N, M, c, t, v).permute(0, 2, 3, 4, 1)

        # prediction
        # (64,400,300,18)
        x = self.fcn(x)
        # output: (64,400,300,18,1)
        output = x.view(N, M, -1, t, v).permute(0, 2, 3, 4, 1)

        return output, feature

class st_gcn(nn.Module):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): (temporal_kernel_size, spatial_kernel_size)
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.

    """
    def __init__(self, 
                in_channels, 
                out_channels, 
                kernel_size, 
                stride = 1, 
                dropout =0, 
                residual = True):
        super().__init__()

        assert len(kernel_size) == 2    # kernel_size should be (temporal, spatial)
        assert kernel_size[0] % 2 == 1 # temporal kernel size should be odd number
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = ConvTemporalGraphical(in_channels, out_channels, kernel_size[1])
        # Only do temporal feature fusion, no channel dimension changed
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,  
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        ) 
        # If no residual connection
        if not residual:
            self.residual = lambda x: 0
        # If residual connection and the output channels doesn't change
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x:x
        # If output channels changed
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x, A):
        res = self.residual(x)
        x, A = self.gcn(x, A)
        # TCN + residual connection
        x = self.tcn(x) + res   # (N, out_channels, T, V)
        return self.relu(x), A



class ConvTemporalGraphical(nn.Module):

    r"""The basic module for applying a spatial graph convolution.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the graph convolving kernel
        t_kernel_size (int): Size of the temporal convolving kernel
        t_stride (int, optional): Stride of the temporal convolution. Default: 1
        t_padding (int, optional): Temporal zero-padding added to both sides of
            the input. Default: 0
        t_dilation (int, optional): Spacing between temporal kernel elements.
            Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output.
            Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes. 
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True):
        super().__init__()

        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias)


    # forward()函数完成图卷积操作,x由（64,2,300,18）变成（64,64,300,18）
    def forward(self, x, A):
        assert A.size(0) == self.kernel_size    # 3
        x = self.conv(x)    #(N,C,T,V)->(N, C*out_channels, T, V)
        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc//self.kernel_size, t, v)# (N, C, out_channels, T, V)
        # 此处的k消失的原因：在k维度上进行了求和操作,也即是x在邻接矩阵A的3个不同的子集上进行乘机操作再进行求和,对应于论文中的公式10
        x = torch.einsum('nkctv,kvw->nctw', (x, A))

        return x.contiguous(), A

        
