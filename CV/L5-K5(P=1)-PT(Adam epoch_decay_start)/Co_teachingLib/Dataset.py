

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


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




