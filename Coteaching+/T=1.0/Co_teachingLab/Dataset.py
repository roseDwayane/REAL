
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class DatasetCoTeachingArray(Dataset):
    def __init__(self, df_x, labels=None, dataset_array=None):
        """
        :param df_x: Dataframe, sub_dataset
        :param labels: np.array, total labels in dataset
        :param dataset_array: np.array, total data in dataset
        """
        self.df_x = df_x
        self.labels = labels
        self.dataset_array = dataset_array

        # 加速
        self.indices = df_x.index.to_numpy()
        self.subscripts_in_dataset = df_x.index.to_numpy()

    def __len__(self):
        return len(self.df_x)

    def __getitem__(self, index):
        return self._single_epoch(index)

    def _single_epoch(self, loc):
        index = self.indices[loc]
        x = self.dataset_array[index]
        label = self.labels[index]

        return x, label


class DatasetCoTeachingArrayBalance(Dataset):
    def __init__(self, df_x, labels=None, dataset_array=None):
        """
        :param df_x: Dataframe, sub_dataset
        :param labels: np.array, total labels in dataset
        :param dataset_array: np.array, total data in dataset
        """
        self.df_x = df_x
        self.labels = labels
        self.dataset_array = dataset_array

        # 加速
        self.indices = df_x.index.to_numpy()
        self.subscripts_in_dataset = df_x.index.to_numpy()

        self.df_x_n = self.df_x[self.df_x['label'] == 0]
        self.df_x_p = self.df_x[self.df_x['label'] == 1]

        self.indices_n = self.df_x_n.index.to_numpy()
        self.indices_p = self.df_x_p.index.to_numpy()

        self.len_n = len(self.indices_n)
        self.len_p = len(self.indices_p)

    def __len__(self):
        # return len(self.df_x)
        return max([self.len_n, self.len_p]) * 2

    def __getitem__(self, index):
        # return self._single_epoch(index)
        if index % 2 == 0:
            return self._single_epoch_n(index // 2)
        else:
            return self._single_epoch_p(index // 2)

    def _single_epoch_n(self, loc):
        if loc >= self.len_n:
            loc = np.random.randint(self.len_n)
        index = self.indices_n[loc]
        x = self.dataset_array[index]
        label = self.labels[index]
        return x, label

    def _single_epoch_p(self, loc):
        if loc >= self.len_p:
            loc = np.random.randint(self.len_p)
        index = self.indices_p[loc]
        x = self.dataset_array[index]
        label = self.labels[index]
        return x, label

