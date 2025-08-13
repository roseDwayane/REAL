
import pandas as pd
import os
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
