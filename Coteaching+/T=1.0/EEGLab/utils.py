
import numpy as np
import pickle

import pandas as pd
import os
import time


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
    #     self.reset()
    #
    # def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def get_value(self):
        return self.avg

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class AccuracyMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt

        self.y_hat = None
        self.y = None
        self.acc = 0.0
        self.sum = 0
        self.count = 0

    def update(self, y_hat, y):
        if self.y is None:
            self.y = y
        else:
            self.y = np.concatenate([self.y, y], axis=0)

        if self.y_hat is None:
            self.y_hat = y_hat
        else:
            self.y_hat = np.concatenate([self.y_hat, y_hat], axis=0)

    def get_value(self):
        y_hat = np.argmax(self.y_hat, axis=-1)
        self.acc = (y_hat == self.y).mean()
        return self.acc

    def get_pred(self):
        y_hat = np.argmax(self.y_hat, axis=-1)
        return (y_hat == self.y).astype('in32')

    def get_proba(self):
        return self._softmax(self.y_hat)

    def _softmax(self, x, axis=-1):
        x_max = np.max(x, axis=axis, keepdims=True)
        x = x - x_max
        x_exp = np.exp(x)
        x_exp_sigma = np.sum(x_exp, axis=axis, keepdims=True)

        return x_exp / x_exp_sigma

    def __str__(self):
        self.get_value()
        fmtstr = '{name} {acc' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)


def mkdir(path):
    import os

    path = path.strip()         # 去除首尾空格
    path = path.rstrip("\\")    # 去除尾部 \ 符号

    if not os.path.exists(path):
        os.makedirs(path)
        return True
    else:
        return False


def adjust_learning_rate(optimizer, init_lr, epoch, epochs):
    """Decay the learning rate based on schedule"""
    import math

    cur_lr = init_lr * 0.5 * (1.0 + math.cos(math.pi * epoch / epochs))
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = init_lr
            # print(init_lr)
        else:
            param_group['lr'] = cur_lr
            # print(cur_lr)
    return cur_lr



