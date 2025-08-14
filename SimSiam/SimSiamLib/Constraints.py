
import torch
import torch.nn as nn
import torch.nn.functional as F


class MaxNorm(object):
    def __init__(self, max_value=1.0, axis=-1, epsilon=1.0e-8):
        self.max_value = max_value
        self.axis = axis
        self.epsilon = epsilon

    def __call__(self, module):
        if hasattr(module, 'weight'):
            # print("Entered for testing!")
            # w=module.weight.data
            # w=w.clamp(0.5,0.7) #将参数范围限制到0.5-0.7之间
            # module.weight.data=w
            w = module.weight.data
            # print(w.shape)
            norms = torch.norm(w, p=2, dim=self.axis, keepdim=True)
            desired = norms.clamp(0, self.max_value)
            w = w * (desired / (self.epsilon + norms))
            module.weight.data = w


class UnitNorm(object):
    def __init__(self, norm_value=0.25, axis=-1, epsilon=1.0e-8):
        self.norm_value = norm_value
        self.axis = axis
        self.epsilon = epsilon

    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            norms = torch.norm(w, p=2, dim=self.axis, keepdim=True)
            w = w / (self.epsilon + norms) * self.norm_value
            module.weight.data = w


class ZeroMean(object):
    def __init__(self, axis=-1):
        self.axis = axis

    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            means = torch.mean(w, dim=self.axis, keepdim=True)
            w = w - means
            module.weight.data = w



