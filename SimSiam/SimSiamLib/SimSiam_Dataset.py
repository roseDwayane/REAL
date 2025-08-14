
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import numpy as np


class DatasetSimSiamCouple(Dataset):
    def __init__(self, x, y=None, dataset_dict=None, samples=251, channels=1, is_test=False):
        """
        :param x: Dataframe
        :param y: 可忽略
        """
        self.df_xy = x
        self.dataset_dict = dataset_dict
        self.samples = samples
        self.channels = channels
        self.labels = torch.tensor(self.df_xy['label'].to_numpy())
        self.is_test = is_test

    def __len__(self):
        return len(self.df_xy) * 512

    def __getitem__(self, index):

        return self._couple_epoch(index)

    def _couple_epoch(self, index):
        """
        EC 与 EO 对比
        :param index:
        :return:
        """

        df_fif_idx = index % len(self.df_xy)
        label = self.labels[df_fif_idx]

        x1 = self._augmentation(df_fif_idx, key_word='self_idx')
        x2 = self._augmentation(df_fif_idx, key_word='couple_idx')

        return (x1, x2), label

    def _augmentation(self, fif_idx, key_word='self_idx'):
        aug_idx = self.df_xy.loc[fif_idx, key_word]
        data = self.dataset_dict[aug_idx]               # torch.tensor() @
        n_times = data.shape[-1]

        sel_offset = np.random.randint(n_times-self.samples)
        x = data[:, :, sel_offset: sel_offset+self.samples]

        return x

    def __repr__(self):
        return self.__class__.__name__


