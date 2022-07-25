import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.clock_driven import functional, layer
__all__ = ['MultiStepCIFAR10Net', 'PConv']

from train import args as parser_args

class PseudoRelu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        if x.requires_grad:
            ctx.save_for_backward(x)
        return torch.relu(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = None
        if ctx.needs_input_grad[0]:
            grad_x = grad_output
        return grad_x, None

pseudoRelu = PseudoRelu.apply

def softThreshold(x, s):
    return torch.sign(x) * torch.relu(torch.abs(x)-s)

def softThresholdinv(x, s):
    return torch.sign(x) * (torch.abs(x) + s)

def softThresholdmod(x, s):
    return torch.sign(x) * pseudoRelu(torch.abs(x)-s)

class PConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        with torch.no_grad():
            if parser_args.sparse_function == 'identity':
                self.mapping = lambda x: x
            elif parser_args.sparse_function == 'st':
                if parser_args.gradual is None:
                    self.mapping = lambda x: softThreshold(x, parser_args.flat_width)
                else:
                    self.mapping = lambda x: x
            elif parser_args.sparse_function == 'stmod':
                if parser_args.gradual is None:
                    self.mapping = lambda x: softThresholdmod(x, parser_args.flat_width)
                else:
                    self.mapping = lambda x: x

    
    def forward(self, x):      
        sparseWeight = self.mapping(self.weight)
        x = F.conv2d(
            x, sparseWeight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x

    @torch.no_grad()
    def getSparsity(self):
        sparseWeight = self.mapping(self.weight)
        temp = sparseWeight.detach().cpu()
        return (temp == 0).sum(), temp.numel()

    @torch.no_grad()
    def getSparseWeight(self):
        return self.mapping(self.weight)
        
    @torch.no_grad()
    def setFlatWidth(self, width):
        if parser_args.sparse_function == 'st':
            self.mapping = lambda x: softThreshold(x, width)
        elif parser_args.sparse_function == 'stmod':
            self.mapping = lambda x: softThresholdmod(x, width)
        
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return PConv(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return PConv(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class VotingLayer(nn.Module):
    def __init__(self, voting_size: int = 10):
        super().__init__()
        self.voting_size = voting_size

    def forward(self, x: torch.Tensor):
        y = F.avg_pool1d(x.unsqueeze(1), self.voting_size, self.voting_size).squeeze(1)
        return y

class CIFAR10Net(nn.Module):
    def __init__(self, channels=256, single_step_neuron: callable = None, **kwargs):
        super().__init__()

        self.conv1 = conv3x3(3, channels)
        self.bn1 = nn.BatchNorm2d(channels)
        self.sn1 = single_step_neuron(**kwargs)

        self.conv2 = conv3x3(channels, channels)
        self.bn2 = nn.BatchNorm2d(channels)
        self.sn2 = single_step_neuron(**kwargs)

        self.conv3 = conv3x3(channels, channels)
        self.bn3 = nn.BatchNorm2d(channels)
        self.sn3 = single_step_neuron(**kwargs)

        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv4 = conv3x3(channels, channels)
        self.bn4 = nn.BatchNorm2d(channels)
        self.sn4 = single_step_neuron(**kwargs)

        self.conv5 = conv3x3(channels, channels)
        self.bn5 = nn.BatchNorm2d(channels)
        self.sn5 = single_step_neuron(**kwargs)

        self.conv6 = conv3x3(channels, channels)
        self.bn6 = nn.BatchNorm2d(channels)
        self.sn6 = single_step_neuron(**kwargs)

        self.pool6 = nn.MaxPool2d(2, 2)

        self.dp7 = layer.Dropout(0.5)
        self.fc7 = conv1x1(channels * 8 * 8, 2048)
        self.sn7 = single_step_neuron(**kwargs)

        self.dp8 = layer.Dropout(0.5)
        self.fc8 = conv1x1(2048, 100)
        self.sn8 = single_step_neuron(**kwargs)
        self.voting = VotingLayer(10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.sn1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.sn2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.sn3(x)
        x = self.pool3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.sn4(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.sn5(x)

        x = self.conv6(x)
        x = self.bn6(x)
        x = self.sn6(x)
        x = self.pool6(x)

        x = x.flatten(1)

        x = self.dp7(x)
        x = self.fc7(x)
        x = self.sn7(x)

        x = self.dp8(x)
        x = self.fc8(x)
        x = self.sn8(x)

        x = self.voting(x)
        return x

class MultiStepCIFAR10Net(CIFAR10Net):
    def __init__(self, channels=256, multi_step_neuron: callable = None, **kwargs):
        super().__init__(channels, multi_step_neuron, **kwargs)
        del self.dp7, self.dp8
        self.dp7 = layer.MultiStepDropout(0.5)
        self.dp8 = layer.MultiStepDropout(0.5)

        self.train_times = 0
        self.epochs = 0


    def forward(self, x: torch.Tensor, T: int):
        x = self.conv1(x)
        x = self.bn1(x)
        x.unsqueeze_(0)
        x = x.repeat(T, 1, 1, 1, 1)

        x = self.sn1(x)

        x = functional.seq_to_ann_forward(x, [self.conv2, self.bn2])
        x = self.sn2(x)

        x = functional.seq_to_ann_forward(x, [self.conv3, self.bn3])
        x = self.sn3(x)

        x = functional.seq_to_ann_forward(x, [self.pool3, self.conv4, self.bn4])
        x = self.sn4(x)

        x = functional.seq_to_ann_forward(x, [self.conv5, self.bn5])
        x = self.sn5(x)

        x = functional.seq_to_ann_forward(x, [self.conv6, self.bn6])
        x = self.sn6(x)
        x = functional.seq_to_ann_forward(x, self.pool6)

        x = x.flatten(2) # [T, N, C, H, W] -> [T, N, CxHxW]

        x = self.dp7(x)

        #### Added for modify from FC to 1x1 Conv
        x = x.unsqueeze(-1)
        x = x.unsqueeze(-1)
        #### [T, N, CxHxW] -> [T, N, CxHxW, 1, 1]

        x = functional.seq_to_ann_forward(x, self.fc7)
        x = self.sn7(x)

        x = self.dp8(x)
        x = functional.seq_to_ann_forward(x, self.fc8)

        #### Added for modify from FC to 1x1 Conv
        x = x.squeeze(-1)
        x = x.squeeze(-1)
        #### [T, N, CxHxW, 1, 1] -> [T, N, CxHxW]
        
        x = self.sn8(x)

        x = functional.seq_to_ann_forward(x, self.voting)
        return x