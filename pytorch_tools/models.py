"""Neural Network Models"""
import math

import torch
import torch.nn as nn
from torch.autograd import Variable


def init_weights_normal(std):
    def wrapper(module):
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
    return wrapper


def init_weights_kaiming(module):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, a=0, mode='fan_in')
    elif isinstance(module, nn.BatchNorm2d):
        num_features = len(module.weight)
        module.weight.data.normal_(mean=0, std=math.sqrt(2. / 9. / num_features)).clamp_(-0.025,0.025)
        nn.init.constant_(module.bias, 0.0)


def init_biases_value(value):
    def wrapper(module):
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            nn.init.constant_(module.bias, val=value)
    return wrapper


class BaseNN(nn.Module):

    @property
    def dummy_input(self):
        dummy_input = Variable(torch.zeros(self.input_dims).unsqueeze(dim=0))
        if self.is_cuda:
            dummy_input = dummy_input.cuda(self.get_device())
        return dummy_input

    @property
    def is_cuda(self):
        return next(self.parameters()).is_cuda

    @property
    def num_params(self):
        return sum([p.numel() for p in self.parameters()])

    @property
    def param_means(self):
        return torch.stack([p.data.view(p.numel()).mean(dim=0)
                            for p in self.parameters()], dim=1)

    @property
    def device(self):
        return next(self.parameters()).device

    def reset(self):
        """Reset model."""
        self.reset_parameters()
        self.zero_grad()
        self.set_dropout_rates(self.init_dropout_rates)

    def reset_parameters(self):
        """Reset all trainable model parameters."""
        for _, module in self.named_modules():
            if (module is not self and
                    getattr(module, "reset_parameters", False)):
                module.reset_parameters()

    def init_zero_grads(self):
        """Initialize all parameter gradients with zero.

        This might be necessary because by default parameters are initialized
        without a corresponding gradient Variable.
        """
        output = self(self.dummy_input)
        output.sum().backward()
        self.zero_grad()

