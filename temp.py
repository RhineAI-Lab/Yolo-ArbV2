from torch import nn
import torch

class SmoothL1LossSr(nn.SmoothL1Loss):
    def __init__(self,smooth_range=1.0):
        super().__init__()
        self.sr = smooth_range

    def __call__(self, p, t):
        sr = self.sr
        loss = super().__call__(p/sr, t/sr)*sr
        return loss

p = torch.Tensor([1.0,2.0,10.0])
t = torch.Tensor([1.0,2.0,3.0])

SL1 = SmoothL1LossSr(2.0)
print(SL1(p,t))



