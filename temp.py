import numpy as np
import torch

SL1 = torch.nn.SmoothL1Loss()

a = torch.Tensor([1,2,3])
b = torch.Tensor([1,3,4])

l = SL1(a,b)
print(l)
