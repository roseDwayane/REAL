
import numpy as np


def softmax(x, axis=1):
    x_max = np.max(x, axis=axis, keepdims=True)
    x = x - x_max
    x_exp = np.exp(x)
    x_exp_sigma = np.sum(x_exp, axis=axis, keepdims=True)

    return x_exp / x_exp_sigma


def transformer_like(x):
    # normalize features
    x_norm = x / np.linalg.norm(x, axis=1, keepdims=True)
    x_conv = np.dot(x_norm, x_norm.T)
    weights = softmax(x_conv, axis=1)

    return np.dot(weights, x)


def add_center(x, normalize=False):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)

    # 特征均值
    Z = np.mean(x, axis=0, keepdims=True)

    # normalize
    Z = Z / np.linalg.norm(Z)

    x = x + Z

    x = x / np.linalg.norm(x, axis=1, keepdims=True)

    return x


class EEGTransformer(object):
    def __init__(self, T=1.0):
        self.T = T

    @staticmethod
    def _softmax(x, axis=-1):
        x_max = np.max(x, axis=axis, keepdims=True)
        x = x - x_max
        x_exp = np.exp(x)
        x_exp_sigma = np.sum(x_exp, axis=axis, keepdims=True)

        return x_exp / x_exp_sigma

    def __call__(self, x, axis=-1):
        x_norm = x / np.linalg.norm(x, axis=1, keepdims=True)
        x_conv = np.dot(x_norm, x_norm.T)
        weights = softmax(x_conv / self.T, axis=axis)

        return np.dot(weights, x)


class SelfAttention(object):
    def __init__(self, T=1.0):
        self.T = T

    @staticmethod
    def _softmax(x, axis=-1):
        x_max = np.max(x, axis=axis, keepdims=True)             # (N, 1)
        x = x - x_max                                           # (N, N)
        x_exp = np.exp(x)                                       # (N, N)
        x_exp_sigma = np.sum(x_exp, axis=axis, keepdims=True)   # (N, 1)

        return x_exp / x_exp_sigma

    def __call__(self, x, axis=-1):
        x_norm = x / np.linalg.norm(x, axis=axis, keepdims=True)    # (N, F)
        x_conv = np.dot(x_norm, x_norm.T)                           # (N, N)
        weights = softmax(x_conv / self.T, axis=axis)               # (N, N)

        y = np.dot(weights, x)                                      # (N, F)
        return y


class SelfAttentionExt(object):
    def __init__(self, T=1.0):
        self.T = T

    @staticmethod
    def _softmax(x, axis=-1):
        x_max = np.max(x, axis=axis, keepdims=True)             # (N, 1)
        x = x - x_max                                           # (N, Nb)
        x_exp = np.exp(x)                                       # (N, Nb)
        sigma_x_exp = np.sum(x_exp, axis=axis, keepdims=True)   # (N, 1)

        return x_exp / sigma_x_exp

    def __call__(self, x, axis=-1, xb=None):
        """

        :param x:   over-sampling 之 epochs 特征矩阵  (N, F)
        :param axis:
        :param xb:  非 over-sampling 之 epochs 特征矩阵 (Nb, F)
        :return:
        """
        if xb is None:
            xb = x
        x_unit = x / np.linalg.norm(x, axis=axis, keepdims=True)        # (N, F)
        xb_unit = xb / np.linalg.norm(xb, axis=axis, keepdims=True)     # (Nb, F)
        x_conv = np.dot(x_unit, xb_unit.T)                              # (N, Nb)
        weights = softmax(x_conv / self.T, axis=axis)                   # (N, Nb)

        y = np.dot(weights, xb)                                         # (N, F)
        return y
