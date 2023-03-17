import numpy as np
import os
import torch.nn as nn
import torch

A = nn.Parameter(torch.rand(1))
B = nn.Parameter(1 - A)
print(A)
print(B)
