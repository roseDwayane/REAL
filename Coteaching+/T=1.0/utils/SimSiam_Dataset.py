
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import numpy as np


class DatasetSimSiam(Dataset):
    def __init__(self, x, y=None, dataset_dict=None, samples=251, channels=1):
        """
        :param x: Dataframe
        :param y: 可忽略
        """
        self.x = x
        self.dataset_dict = dataset_dict
        self.samples = samples
        self.channels = channels
        self.labels = torch.tensor(self.x['label'].to_numpy())

    def __len__(self):
        return len(self.x) * 512

    def __getitem__(self, index):
        if self.channels == 1:
            return self._single_epoch(index)
        else:
            return self._double_epochs(index)

    def _single_epoch(self, index):
        fif_idx = index % len(self.x)
        label = self.labels[fif_idx]
        fif_data = self.dataset_dict[fif_idx]   # torch.tensor() @
        n_times = fif_data.shape[-1]
        # print(fif_data.shape)   # (1, 1, 30, n_times)

        # sel_offset = np.random.randint(n_times-self.duration, size=2)
        # x1 = fif_data[:, :, sel_offset[0]: sel_offset[0]+self.duration]
        # x2 = fif_data[:, :, sel_offset[1]: sel_offset[1]+self.duration]

        sel_offset1 = np.random.randint(n_times//2)
        x1 = fif_data[:, :, sel_offset1: sel_offset1 + self.samples]

        sel_offset2 = np.random.randint(n_times//2, n_times-self.samples)
        x2 = fif_data[:, :, sel_offset2: sel_offset2 + self.samples]

        # x1 = torch.tensor(x1)
        # x2 = torch.tensor(x2)
        # label = torch.tensor(label)

        return (x1, x2), label

    def _double_epochs(self, index):
        fif_idx = index % len(self.x)
        label = self.labels[fif_idx]
        fif_data = self.dataset_dict[fif_idx]
        n_times = fif_data.shape[-1]
        # print(fif_data.shape)   # (1, 30, n_times)

        # sel_offset = np.random.randint(self.duration//2, size=self.channels)
        sel_offset = np.random.randint(n_times-self.samples, size=self.channels * 2)
        x1 = fif_data[:, :, sel_offset[0]: sel_offset[0]+self.samples]
        x2 = fif_data[:, :, sel_offset[1]: sel_offset[1]+self.samples]
        x3 = fif_data[:, :, sel_offset[2]: sel_offset[2]+self.samples]
        x4 = fif_data[:, :, sel_offset[3]: sel_offset[3]+self.samples]

        # x1 = np.concatenate([x1, x3], axis=-2)
        # x2 = np.concatenate([x2, x4], axis=-2)

        x1 = torch.concat([x1, x3], dim=-2)
        x2 = torch.concat([x2, x4], dim=-2)

        # x1 = torch.tensor(x1)
        # x2 = torch.tensor(x2)

        # x1 = torch.from_numpy(x1)
        # x2 = torch.from_numpy(x2)
        # label = torch.tensor(label)

        return (x1, x2), label


class DatasetSimSiamMixup(Dataset):
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
        if self.is_test:
            return self._mixup_epoch_self(index)
        else:
            return self._mixup_couple_epoch(index)

    def _mixup_couple_epoch(self, index):
        """
        EC 与 EO 混合
        :param index:
        :return:
        """

        fif_idx = index % len(self.df_xy)
        label = self.labels[fif_idx]
        fif_data = self.dataset_dict[fif_idx]   # torch.tensor() @
        n_times = fif_data.shape[-1]

        fif_idx_couple = self.df_xy.loc[fif_idx, 'couple_idx']
        fif_data_couple = self.dataset_dict[fif_idx_couple]   # torch.tensor() @
        n_times_couple = fif_data_couple.shape[-1]

        sel_offset = np.random.randint(n_times//2, n_times-self.samples, size=2)
        x1 = fif_data[:, :, sel_offset[0]: sel_offset[0]+self.samples]
        x2 = fif_data[:, :, sel_offset[1]: sel_offset[1]+self.samples]

        sel_offset_couple = np.random.randint(n_times_couple//2, n_times_couple-self.samples, size=2)
        x3 = fif_data_couple[:, :, sel_offset_couple[0]: sel_offset_couple[0]+self.samples]
        x4 = fif_data_couple[:, :, sel_offset_couple[1]: sel_offset_couple[1]+self.samples]

        alpha = np.random.rand(2) * 0.25     # [0, 0.1)

        x5 = self._interpolate(x1, x3, beta=alpha[0])
        x6 = self._interpolate(x2, x4, beta=alpha[1])

        return (x5, x6), label

    def _mixup_epoch_self(self, index):
        fif_idx = index % len(self.df_xy)
        label = self.labels[fif_idx]
        fif_data = self.dataset_dict[fif_idx]   # torch.tensor() @
        n_times = fif_data.shape[-1]
        # print(fif_data.shape)   # (1, 1, 30, n_times)

        sel_offset = np.random.randint(n_times//2, n_times-self.samples, size=4)
        x1 = fif_data[:, :, sel_offset[0]: sel_offset[0]+self.samples]
        x2 = fif_data[:, :, sel_offset[1]: sel_offset[1]+self.samples]
        x3 = fif_data[:, :, sel_offset[2]: sel_offset[2]+self.samples]
        x4 = fif_data[:, :, sel_offset[3]: sel_offset[3]+self.samples]

        alpha = np.random.rand(2) * 0.25     # [0, 0.1)

        x5 = self._interpolate(x1, x3, beta=alpha[0])
        x6 = self._interpolate(x2, x4, beta=alpha[1])

        return (x5, x6), label

    def _interpolate(self, x, y, beta=0.5):
        return x * beta + y * (1 - beta)

    def _mixup_epoch_2(self, index):
        fif_idx = index % len(self.df_xy)
        label = self.labels[fif_idx]
        fif_data = self.dataset_dict[fif_idx]   # torch.tensor() @
        n_times = fif_data.shape[-1]
        # print(fif_data.shape)   # (1, 1, 30, n_times)

        sel_offset1 = np.random.randint(n_times//2, size=2)
        x1 = fif_data[:, :, sel_offset1[0]: sel_offset1[0]+self.samples]
        x2 = fif_data[:, :, sel_offset1[1]: sel_offset1[1]+self.samples]

        sel_offset2 = np.random.randint(n_times//2, n_times-self.samples, size=2)
        x3 = fif_data[:, :, sel_offset2[0]: sel_offset2[0]+self.samples]
        x4 = fif_data[:, :, sel_offset2[1]: sel_offset2[1]+self.samples]

        x5 = (x1 + x3) * 0.5
        x6 = (x2 + x4) * 0.5

        return (x5, x6), label

    def _mixup_epoch(self, index):
        fif_idx = index % len(self.df_xy)
        label = self.labels[fif_idx]
        fif_data = self.dataset_dict[fif_idx]   # torch.tensor() @
        n_times = fif_data.shape[-1]
        # print(fif_data.shape)   # (1, 1, 30, n_times)

        sel_offset1 = np.random.randint(n_times//2, size=2)
        x1 = fif_data[:, :, sel_offset1[0]: sel_offset1[0]+self.samples]
        x2 = fif_data[:, :, sel_offset1[1]: sel_offset1[1]+self.samples]

        sel_offset2 = np.random.randint(n_times//2, n_times-self.samples, size=2)
        x3 = fif_data[:, :, sel_offset2[0]: sel_offset2[0]+self.samples]
        x4 = fif_data[:, :, sel_offset2[1]: sel_offset2[1]+self.samples]

        alpha = np.random.rand(2) * 0.1     # [0, 0.1)

        # x5 = (x1 + x3) * 0.5
        # x6 = (x2 + x4) * 0.5

        x5 = self._interpolate(x1, x3, beta=alpha[0])
        x6 = self._interpolate(x2, x4, beta=alpha[1])

        return (x5, x6), label


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

        return  self._couple_epoch(index)

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


class DatasetSimSiamCoupleBalance(Dataset):
    def __init__(self, x, y=None, dataset_dict=None, samples=251, channels=1, is_test=False):
        """
        :param x: Dataframe
        :param y: 可忽略
        """
        self.df_xy = x
        self.dataset_dict = dataset_dict
        self.samples = samples
        self.channels = channels
        self.is_test = is_test

        self.labels = torch.tensor(self.df_xy['label'].to_numpy())

        self.df_n = self.df_xy[self.df_xy['label'] == 0]
        self.df_p = self.df_xy[self.df_xy['label'] == 1]

        self.n_negative = len(self.df_n)
        self.n_positive = len(self.df_p)

    def __len__(self):
        return len(self.df_xy) * 512

    def __getitem__(self, index):
        if index % 2 == 0:
            return self._couple_epoch_negative(index // 2)
        else:
            return self._couple_epoch_positive(index // 2)

    def _couple_epoch_negative(self, idx):
        """
        EC 与 EO 对比
        :param idx: index // 2
        :return:
        """

        # df_fif_idx = index % len(self.df_xy)
        pos_n = idx % self.n_negative
        df_fif_idx = self.df_n.index[pos_n]

        label = self.labels[df_fif_idx]

        x1 = self._crop(df_fif_idx, key_word='self_idx')
        x2 = self._crop(df_fif_idx, key_word='couple_idx')

        return (x1, x2), label

    def _couple_epoch_positive(self, idx):
        """
        EC 与 EO 对比
        :param idx: index // 2
        :return:
        """

        # df_fif_idx = index % len(self.df_xy)
        pos_p = idx % self.n_positive
        df_fif_idx = self.df_p.index[pos_p]

        label = self.labels[df_fif_idx]

        x1 = self._crop(df_fif_idx, key_word='self_idx')
        x2 = self._crop(df_fif_idx, key_word='couple_idx')

        return (x1, x2), label

    def _crop(self, fif_idx, key_word='self_idx'):
        aug_idx = self.df_xy.loc[fif_idx, key_word]
        data = self.dataset_dict[aug_idx]               # torch.tensor() @
        n_times = data.shape[-1]

        sel_offset = np.random.randint(n_times-self.samples)
        x = data[:, :, sel_offset: sel_offset+self.samples]

        return x

    def __repr__(self):
        return self.__class__.__name__


class DatasetSimSiamCoupleRandom(Dataset):
    def __init__(self, x, y=None,
                 dataset_dict=None,
                 samples=251,
                 channels=1,
                 couple_sel_p=0.5,
                 is_test=False):
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

        self.p = 0.10
        self.q = 0.10
        self.r = 1.0e-6

        self.rand_array = None
        self._gen_rand_array()

        self.couple_sel_p = couple_sel_p

    def __len__(self):
        return len(self.df_xy) * 512

    def __getitem__(self, index):
        if index == 0:
            self._gen_rand_array()

        if self.is_test:
            return self._mixup_epoch_self(index)
        else:
            return self._couple_epoch_amp(index)

    def _couple_epoch_amp(self, index):
        """
        EC 与 EO 对比
        :param index:
        :return:
        """

        df_fif_idx = index % len(self.df_xy)
        label = self.labels[df_fif_idx]

        x1 = self._augmentation(df_fif_idx, key_word='self_idx')

        if self.rand_array[index] < self.couple_sel_p:
            x2 = self._augmentation(df_fif_idx, key_word='couple_idx')
        else:
            x2 = self._augmentation(df_fif_idx, key_word='self_idx')

        return (x1, x2), label

        # amp, noise = self._amp()
        #
        # x5 = x1 * amp[0] + noise[0]
        # x6 = x2 * amp[1] + noise[1]
        #
        # return (x5, x6), label

    def _augmentation(self, fif_idx, key_word='self_idx'):
        aug_idx = self.df_xy.loc[fif_idx, key_word]
        data = self.dataset_dict[aug_idx]               # torch.tensor() @
        n_times = data.shape[-1]

        sel_offset = np.random.randint(n_times-self.samples)
        x = data[:, :, sel_offset: sel_offset+self.samples]

        return x

    def _amp(self):
        # 以q的概率衰减幅值
        n = torch.rand((2, 1, 30, 1))
        a = n * 0.25 + 0.75     # [0.75, 1.0)
        a[n > self.q] = 1.0

        # 以p的概率翻转极性
        n = torch.rand((2, 1, 30, 1))
        s = torch.ones((2, 1, 30, 1))
        s[n < self.p] = -1.0

        noise = torch.randn((2, 1, 30, self.samples)) * self.r

        return a * s, noise

    def _gen_rand_array(self):
        self.rand_array = np.random.rand(self.__len__())

    def _mixup_epoch_self(self, index):
        fif_idx = index % len(self.df_xy)
        label = self.labels[fif_idx]
        fif_data = self.dataset_dict[fif_idx]   # torch.tensor() @
        n_times = fif_data.shape[-1]
        # print(fif_data.shape)   # (1, 1, 30, n_times)

        sel_offset = np.random.randint(n_times-self.samples, size=4)
        x1 = fif_data[:, :, sel_offset[0]: sel_offset[0]+self.samples]
        x2 = fif_data[:, :, sel_offset[1]: sel_offset[1]+self.samples]
        x3 = fif_data[:, :, sel_offset[2]: sel_offset[2]+self.samples]
        x4 = fif_data[:, :, sel_offset[3]: sel_offset[3]+self.samples]

        alpha = np.random.rand(2) * 0.25     # [0, 0.1)

        x5 = self._interpolate(x1, x3, beta=alpha[0])
        x6 = self._interpolate(x2, x4, beta=alpha[1])

        return (x5, x6), label

    def _interpolate(self, x, y, beta=0.5):
        return x * beta + y * (1 - beta)

    def __repr__(self):
        return self.__class__.__name__ + 'p={0: %.2f}, q={1: %.2f}, r={2: %.2e}'.format(self.p, self.q, self.r)


class DatasetSimSiamCoupleAmp(Dataset):
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

        self.p = 0.10
        self.q = 0.10
        self.r = 1.0e-6

        self.couple_p = None
        self._gen_sel_p()

        # self.q = 0.
        # self.r = 0.

    def __len__(self):
        return len(self.df_xy) * 512

    def __getitem__(self, index):
        if self.is_test:
            return self._mixup_epoch_self(index)
        else:
            return self._couple_epoch_amp(index)

    def _couple_epoch_amp(self, index):
        """
        EC 与 EO 对比
        :param index:
        :return:
        """

        df_fif_idx = index % len(self.df_xy)
        label = self.labels[df_fif_idx]

        x1 = self._augmentation(df_fif_idx, key_word='self_idx')
        x2 = self._augmentation(df_fif_idx, key_word='couple_idx')

        amp, noise = self._amp()

        x5 = x1 * amp[0] + noise[0]
        x6 = x2 * amp[1] + noise[1]

        return (x5, x6), label

    def _augmentation(self, fif_idx, key_word='self_idx'):
        aug_idx = self.df_xy.loc[fif_idx, key_word]
        data = self.dataset_dict[aug_idx]               # torch.tensor() @
        n_times = data.shape[-1]

        sel_offset = np.random.randint(n_times-self.samples)
        x = data[:, :, sel_offset: sel_offset+self.samples]

        return x

    def _amp(self):
        # 以q的概率衰减幅值
        n = torch.rand((2, 1, 30, 1))
        a = n * 0.25 + 0.75     # [0.75, 1.0)
        a[n > self.q] = 1.0

        # 以p的概率翻转极性
        n = torch.rand((2, 1, 30, 1))
        s = torch.ones((2, 1, 30, 1))
        s[n < self.p] = -1.0

        noise = torch.randn((2, 1, 30, self.samples)) * self.r

        return a * s, noise

    def _gen_sel_p(self):
        self.couple_p =  np.random.rand(self.__len__())

    def _mixup_epoch_self(self, index):
        fif_idx = index % len(self.df_xy)
        label = self.labels[fif_idx]
        fif_data = self.dataset_dict[fif_idx]   # torch.tensor() @
        n_times = fif_data.shape[-1]
        # print(fif_data.shape)   # (1, 1, 30, n_times)

        sel_offset = np.random.randint(n_times-self.samples, size=4)
        x1 = fif_data[:, :, sel_offset[0]: sel_offset[0]+self.samples]
        x2 = fif_data[:, :, sel_offset[1]: sel_offset[1]+self.samples]
        x3 = fif_data[:, :, sel_offset[2]: sel_offset[2]+self.samples]
        x4 = fif_data[:, :, sel_offset[3]: sel_offset[3]+self.samples]

        alpha = np.random.rand(2) * 0.25     # [0, 0.1)

        x5 = self._interpolate(x1, x3, beta=alpha[0])
        x6 = self._interpolate(x2, x4, beta=alpha[1])

        return (x5, x6), label

    def _interpolate(self, x, y, beta=0.5):
        return x * beta + y * (1 - beta)

    def __repr__(self):
        return self.__class__.__name__ + 'p={0: %.2f}, q={1: %.2f}, r={2: %.2e}'.format(self.p, self.q, self.r)


class DatasetSimSiamCoupleDisturb(Dataset):
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

        self.p = 0.10
        self.q = 0.10
        self.r = 1.0e-6

        # self.couple_p = None
        # self._gen_sel_p()

    def __len__(self):
        return len(self.df_xy) * 512

    def __getitem__(self, index):
        return self._couple_epochs(index)

    def _couple_epochs(self, index):
        """
        EC 与 EO 对比
        :param index:
        :return:
        """

        df_fif_idx = index % len(self.df_xy)
        label = self.labels[df_fif_idx]

        x1 = self._crop(df_fif_idx, key_word='self_idx')
        x2 = self._crop(df_fif_idx, key_word='couple_idx')

        amp, noise = self._disturb()

        x5 = x1 * amp[0] + noise[0]
        x6 = x2 * amp[1] + noise[1]

        return (x5, x6), label

    def _crop(self, fif_idx, key_word='self_idx'):
        aug_idx = self.df_xy.loc[fif_idx, key_word]
        data = self.dataset_dict[aug_idx]               # torch.tensor(): (1, 30, n_times)
        n_times = data.shape[-1]

        sel_offset = np.random.randint(n_times-self.samples)
        x = data[:, :, sel_offset: sel_offset+self.samples]

        return x

    def _disturb(self):
        # 以q的概率衰减幅值
        n = torch.rand((2, 1, 30, 1))
        a = n * 0.25 + 0.75     # [0.75, 1.0)
        a[n > self.q] = 1.0

        # 以p的概率翻转极性
        n = torch.rand((2, 1, 30, 1))
        s = torch.ones((2, 1, 30, 1))
        s[n < self.p] = -1.0

        noise = torch.randn((2, 1, 30, self.samples)) * self.r

        return a * s, noise

    def _gen_sel_p(self):
        self.couple_p =  np.random.rand(self.__len__())

    def _mixup_epoch_self(self, index):
        fif_idx = index % len(self.df_xy)
        label = self.labels[fif_idx]
        fif_data = self.dataset_dict[fif_idx]   # torch.tensor() @
        n_times = fif_data.shape[-1]
        # print(fif_data.shape)   # (1, 1, 30, n_times)

        sel_offset = np.random.randint(n_times-self.samples, size=4)
        x1 = fif_data[:, :, sel_offset[0]: sel_offset[0]+self.samples]
        x2 = fif_data[:, :, sel_offset[1]: sel_offset[1]+self.samples]
        x3 = fif_data[:, :, sel_offset[2]: sel_offset[2]+self.samples]
        x4 = fif_data[:, :, sel_offset[3]: sel_offset[3]+self.samples]

        alpha = np.random.rand(2) * 0.25     # [0, 0.1)

        x5 = self._interpolate(x1, x3, beta=alpha[0])
        x6 = self._interpolate(x2, x4, beta=alpha[1])

        return (x5, x6), label

    def _interpolate(self, x, y, beta=0.5):
        return x * beta + y * (1 - beta)

    def __repr__(self):
        return self.__class__.__name__ + 'p={0: %.2f}, q={1: %.2f}, r={2: %.2e}'.format(self.p, self.q, self.r)


class DatasetSimSiamDoubleCouple(Dataset):
    def __init__(self, x, y=None,
                 dataset_dict=None,
                 samples=251,
                 channels=2,
                 is_test=False):
        """
        :param x: Dataframe
        :param y: 可忽略
        """
        self.df_xy = x
        self.dataset_dict = dataset_dict
        self.samples = samples
        self.channels = channels
        self.is_test = is_test

        self.labels = torch.tensor(self.df_xy['label'].to_numpy())

        self.df_n = self.df_xy[self.df_xy['label'] == 0]
        self.df_p = self.df_xy[self.df_xy['label'] == 1]

        self.n_negative = len(self.df_n)
        self.n_positive = len(self.df_p)

    def __len__(self):
        return len(self.df_xy) * 512

    def __getitem__(self, index):
        if index % 2 == 0:
            return self._couple_epoch_negative(index // 2)
        else:
            return self._couple_epoch_positive(index // 2)

    def _couple_epoch_negative(self, idx):
        """
        EC 与 EO 对比
        :param idx: index // 2
        :return:
        """

        # df_fif_idx = index % len(self.df_xy)
        pos_n = idx % self.n_negative
        df_fif_idx = self.df_n.index[pos_n]

        label = self.labels[df_fif_idx]

        x1 = self._crop(df_fif_idx, key_word='self_idx')
        x2 = self._crop(df_fif_idx, key_word='couple_idx')

        return (x1, x2), label

    def _couple_epoch_positive(self, idx):
        """
        EC 与 EO 对比
        :param idx: index // 2
        :return:
        """

        # df_fif_idx = index % len(self.df_xy)
        pos_p = idx % self.n_positive
        df_fif_idx = self.df_p.index[pos_p]

        label = self.labels[df_fif_idx]

        x1 = self._crop(df_fif_idx, key_word='self_idx')
        x2 = self._crop(df_fif_idx, key_word='couple_idx')

        return (x1, x2), label

    def _crop(self, fif_idx, key_word='self_idx'):
        aug_idx = self.df_xy.loc[fif_idx, key_word]
        data = self.dataset_dict[aug_idx]               # torch.tensor() @
        n_times = data.shape[-1]

        sel_offset = np.random.randint(0, n_times-self.samples, self.channels)

        x_list = []
        for ch in range(self.channels):
            x = data[:, :, sel_offset[ch]: sel_offset[ch]+self.samples]
            x_list.append(x)

        return torch.cat(x_list, dim=1)

    def __repr__(self):
        return self.__class__.__name__


class DatasetSimSiamBalance(Dataset):
    def __init__(self, x, y=None,
                 dataset_dict=None,
                 samples=251,
                 channels=1,
                 p=0.25, q=0.25, r=1.0e-5,
                 is_test=False):
        """
        :param x: Dataframe
        :param y: 可忽略
        """
        self.df_xy = x
        self.dataset_dict = dataset_dict
        self.samples = samples
        self.channels = channels
        self.is_test = is_test

        self.labels = torch.tensor(self.df_xy['label'].to_numpy())

        self.df_negative = self.df_xy[self.df_xy['label'] == 0]
        self.df_positive = self.df_xy[self.df_xy['label'] == 1]

        self.len_negative = len(self.df_negative)
        self.len_positive = len(self.df_positive)

        self.p = p
        self.q = q
        self.r = r

    def __len__(self):
        return len(self.df_xy) * 512

    def __getitem__(self, index):
        if index % 2 == 0:
            return self._epochs_negative(index // 2)
        else:
            return self._epochs_positive(index // 2)

    def _epochs_negative(self, idx):
        """
        EC 与 EC 对比
        :param idx: index // 2
        :return:
        """

        # df_fif_idx = index % len(self.df_xy)
        pos_n = idx % self.len_negative
        df_fif_idx = self.df_negative.index[pos_n]

        label = self.labels[df_fif_idx]

        x1 = self._crop(df_fif_idx, col='self_idx')
        x2 = self._crop(df_fif_idx, col='self_idx')

        amp, noise = self._disturb()

        x5 = x1 * amp[0] + noise[0]
        x6 = x2 * amp[1] + noise[1]

        return (x5, x6), label

    def _epochs_positive(self, idx):
        """
        EC 与 EC 对比
        :param idx: index // 2
        :return:
        """

        # df_fif_idx = index % len(self.df_xy)
        pos_p = idx % self.len_positive
        df_fif_idx = self.df_positive.index[pos_p]

        label = self.labels[df_fif_idx]

        x1 = self._crop(df_fif_idx, col='self_idx')
        x2 = self._crop(df_fif_idx, col='self_idx')

        amp, noise = self._disturb()

        x5 = x1 * amp[0] + noise[0]
        x6 = x2 * amp[1] + noise[1]

        return (x5, x6), label

    def _crop(self, fif_idx, col='self_idx'):
        aug_idx = self.df_xy.loc[fif_idx, col]
        data = self.dataset_dict[aug_idx]               # torch.tensor() @
        n_times = data.shape[-1]

        sel_offset = np.random.randint(n_times-self.samples)
        x = data[:, :, sel_offset: sel_offset+self.samples]

        return x

    def _disturb(self):
        # 以q的概率衰减幅值
        n = torch.rand((2, 1, 30, 1))
        a = n * 0.25 + 0.75     # [0.75, 1.0)
        a[n > self.q] = 1.0

        # 以p的概率翻转极性
        n = torch.rand((2, 1, 30, 1))
        s = torch.ones((2, 1, 30, 1))
        s[n < self.p] = -1.0

        noise = torch.randn((2, 1, 30, self.samples)) * self.r

        return a * s, noise

    def __repr__(self):
        return self.__class__.__name__


class DatasetSimSiamSelf(Dataset):
    def __init__(self, x, y=None,
                 dataset_dict=None,
                 samples=251,
                 channels=1,
                 p=0.25, q=0.25, r=1.0e-5,
                 is_test=False):
        """
        :param x: Dataframe
        :param y: 可忽略
        """
        self.df_xy = x
        self.dataset_dict = dataset_dict
        self.samples = samples
        self.channels = channels
        self.is_test = is_test

        self.labels = torch.tensor(self.df_xy['label'].to_numpy())

        self.df_negative = self.df_xy[self.df_xy['label'] == 0]
        self.df_positive = self.df_xy[self.df_xy['label'] == 1]

        self.len_negative = len(self.df_negative)
        self.len_positive = len(self.df_positive)

        self.p = p
        self.q = q
        self.r = r

    def __len__(self):
        return len(self.df_xy) * 512

    def __getitem__(self, index):
        if index % 2 == 0:
            return self._epochs_negative(index // 2)
        else:
            return self._epochs_positive(index // 2)

    def _epochs_negative(self, idx):
        """
        EC 与 EC 对比
        :param idx: index // 2
        :return:
        """

        pos_n = idx % self.len_negative
        df_fif_idx = self.df_negative.index[pos_n]

        label = self.labels[df_fif_idx]

        x1 = self._crop(df_fif_idx, col='self_idx')
        x2 = self._crop(df_fif_idx, col='self_idx')

        amp, noise = self._disturb()

        x5 = x1 * amp[0] + noise[0]
        x6 = x2 * amp[1] + noise[1]

        return (x5, x6), label

    def _epochs_positive(self, idx):
        """
        EC 与 EC 对比
        :param idx: index // 2
        :return:
        """

        # df_fif_idx = index % len(self.df_xy)
        pos_p = idx % self.len_positive
        df_fif_idx = self.df_positive.index[pos_p]

        label = self.labels[df_fif_idx]

        x1 = self._crop(df_fif_idx, col='self_idx')
        x2 = self._crop(df_fif_idx, col='self_idx')

        amp, noise = self._disturb()

        x5 = x1 * amp[0] + noise[0]
        x6 = x2 * amp[1] + noise[1]

        return (x5, x6), label

    def _crop(self, fif_idx, col='self_idx'):
        aug_idx = self.df_xy.loc[fif_idx, col]
        data = self.dataset_dict[aug_idx]               # torch.tensor() @
        n_times = data.shape[-1]

        sel_offset = np.random.randint(n_times-self.samples)
        x = data[:, :, sel_offset: sel_offset+self.samples]

        return x

    def _disturb(self):
        # 以q的概率衰减幅值
        n = torch.rand((2, 1, 30, 1))
        a = n * 0.25 + 0.75     # [0.75, 1.0)
        a[n > self.q] = 1.0

        # 以p的概率翻转极性
        n = torch.rand((2, 1, 30, 1))
        s = torch.ones((2, 1, 30, 1))
        s[n < self.p] = -1.0

        noise = torch.randn((2, 1, 30, self.samples)) * self.r

        return a * s, noise

    def __repr__(self):
        return self.__class__.__name__


class DatasetSimSiamAmp(Dataset):
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
        if self.is_test:
            return self._epoch_end(index)
        else:
            return self._epoch_amp(index)

    def _epoch_amp(self, index):
        """
        随机调整每个channel的'幅值'与'极性'
        :param index:
        :return:
        """

        fif_idx = index % len(self.df_xy)
        label = self.labels[fif_idx]
        fif_data = self.dataset_dict[fif_idx]   # torch.tensor() @
        n_times = fif_data.shape[-1]

        sel_offset = np.random.randint(n_times-self.samples, size=2)
        x1 = fif_data[:, :, sel_offset[0]: sel_offset[0]+self.samples]
        x2 = fif_data[:, :, sel_offset[1]: sel_offset[1]+self.samples]

        n = torch.rand((2, 1, 30, 1))
        a = n * 0.25 + 0.75     # [0.75, 1.0)
        s = torch.sign(n - 0.5)

        # amp = torch.tensor(amp)
        # s = torch.tensor(s)

        x5 = x1 * a[0] * s[0]
        x6 = x2 * a[1] * s[1]

        return (x5, x6), label

    def _epoch_end(self, index):
        fif_idx = index % len(self.df_xy)
        label = self.labels[fif_idx]
        fif_data = self.dataset_dict[fif_idx]   # torch.tensor() @
        n_times = fif_data.shape[-1]

        sel_offset = np.random.randint(n_times//2, n_times-self.samples, size=2)
        x1 = fif_data[:, :, sel_offset[0]: sel_offset[0]+self.samples]
        x2 = fif_data[:, :, sel_offset[1]: sel_offset[1]+self.samples]

        return (x1, x2), label


class DatasetSimSiamAmpTail(Dataset):
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
        if self.is_test:
            return self._epoch_end(index)
        else:
            return self._epoch_amp(index)

    def _epoch_amp(self, index):
        """
        随机调整每个channel的'幅值'与'极性'
        :param index:
        :return:
        """

        fif_idx = index % len(self.df_xy)
        label = self.labels[fif_idx]
        fif_data = self.dataset_dict[fif_idx]   # torch.tensor() @
        n_times = fif_data.shape[-1]

        # sel_offset = np.random.randint(n_times-self.samples, size=2)
        # 2nd half only
        sel_offset = np.random.randint(n_times//2, n_times-self.samples, size=2)
        x1 = fif_data[:, :, sel_offset[0]: sel_offset[0]+self.samples]
        x2 = fif_data[:, :, sel_offset[1]: sel_offset[1]+self.samples]

        amp = self._amp()

        x5 = x1 * amp[0]
        x6 = x2 * amp[1]

        return (x5, x6), label

    def _amp(self):
        # 以q的概率衰减幅值
        q = 0.25
        n = torch.rand((2, 1, 30, 1))
        a = n * 0.25 + 0.75     # [0.75, 1.0)
        a[n > q] = 1.0

        # 以p的概率翻转极性
        p = 0.25
        n = torch.rand((2, 1, 30, 1))
        s = torch.ones((2, 1, 30, 1))
        s[n < p] = -1.0

        return a * s

    def _epoch_end(self, index):
        fif_idx = index % len(self.df_xy)
        label = self.labels[fif_idx]
        fif_data = self.dataset_dict[fif_idx]   # torch.tensor() @
        n_times = fif_data.shape[-1]

        sel_offset = np.random.randint(n_times//2, n_times-self.samples, size=2)
        x1 = fif_data[:, :, sel_offset[0]: sel_offset[0]+self.samples]
        x2 = fif_data[:, :, sel_offset[1]: sel_offset[1]+self.samples]

        return (x1, x2), label


class DatasetSimSiamAmp2(Dataset):
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
        if self.is_test:
            return self._epoch_end(index)
        else:
            return self._epoch_amp(index)

    def _epoch_amp(self, index):
        """
        随机调整每个channel的'幅值'与'极性'
        :param index:
        :return:
        """

        fif_idx = index % len(self.df_xy)
        label = self.labels[fif_idx]
        fif_data = self.dataset_dict[fif_idx]   # torch.tensor() @
        n_times = fif_data.shape[-1]

        sel_offset = np.random.randint(n_times-self.samples, size=2)
        x1 = fif_data[:, :, sel_offset[0]: sel_offset[0]+self.samples]
        x2 = fif_data[:, :, sel_offset[1]: sel_offset[1]+self.samples]

        amp = self._amp()

        x5 = x1 * amp[0]
        x6 = x2 * amp[1]

        return (x5, x6), label

    def _amp(self):
        # 以q的概率衰减幅值
        q = 0.25
        n = torch.rand((2, 1, 30, 1))
        a = n * 0.25 + 0.75     # [0.75, 1.0)
        a[n > q] = 1.0

        # 以p的概率翻转极性
        p = 0.25
        n = torch.rand((2, 1, 30, 1))
        s = torch.ones((2, 1, 30, 1))
        s[n < p] = -1.0

        return a * s

    def _epoch_end(self, index):
        fif_idx = index % len(self.df_xy)
        label = self.labels[fif_idx]
        fif_data = self.dataset_dict[fif_idx]   # torch.tensor() @
        n_times = fif_data.shape[-1]

        sel_offset = np.random.randint(n_times//2, n_times-self.samples, size=2)
        x1 = fif_data[:, :, sel_offset[0]: sel_offset[0]+self.samples]
        x2 = fif_data[:, :, sel_offset[1]: sel_offset[1]+self.samples]

        return (x1, x2), label


class DatasetSimSiamAmp3(Dataset):
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
        if self.is_test:
            return self._epoch_end(index)
        else:
            return self._epoch_amp(index)

    def _epoch_amp(self, index):
        """
        随机调整每个channel的'幅值'与'极性'
        :param index:
        :return:
        """

        fif_idx = index % len(self.df_xy)
        label = self.labels[fif_idx]
        fif_data = self.dataset_dict[fif_idx]   # torch.tensor() @
        n_times = fif_data.shape[-1]

        sel_offset = np.random.randint(n_times-self.samples, size=2)
        x1 = fif_data[:, :, sel_offset[0]: sel_offset[0]+self.samples]
        x2 = fif_data[:, :, sel_offset[1]: sel_offset[1]+self.samples]

        amp, noise = self._amp()

        x5 = x1 * amp[0] + noise[0]
        x6 = x2 * amp[1] + noise[1]

        return (x5, x6), label

    def _amp(self):
        # 以q的概率衰减幅值
        q = 0.25
        n = torch.rand((2, 1, 30, 1))
        a = n * 0.25 + 0.75     # [0.75, 1.0)
        a[n > q] = 1.0

        # 以p的概率翻转极性
        p = 0.25
        n = torch.rand((2, 1, 30, 1))
        s = torch.ones((2, 1, 30, 1))
        s[n < p] = -1.0

        noise = torch.randn((2, 1, 30, 1)) * 1.0e-4

        return a * s, noise

    def _epoch_end(self, index):
        fif_idx = index % len(self.df_xy)
        label = self.labels[fif_idx]
        fif_data = self.dataset_dict[fif_idx]   # torch.tensor() @
        n_times = fif_data.shape[-1]

        sel_offset = np.random.randint(n_times//2, n_times-self.samples, size=2)
        x1 = fif_data[:, :, sel_offset[0]: sel_offset[0]+self.samples]
        x2 = fif_data[:, :, sel_offset[1]: sel_offset[1]+self.samples]

        return (x1, x2), label


class DatasetSimSiamAmp3M(Dataset):
    def __init__(self, x, y=None,
                 dataset_dict=None,
                 samples=251, channels=1,
                 p=0.25, q=0.25, r=1.0e-5,
                 is_test=False):
        """
        修改了 noise 的 bug.
        :param x: Dataframe
        :param y: 可忽略
        """
        self.df_xy = x
        self.dataset_dict = dataset_dict
        self.samples = samples
        self.channels = channels
        self.labels = torch.tensor(self.df_xy['label'].to_numpy())
        self.is_test = is_test

        self.p = p
        self.q = q
        self.r = r

    def __len__(self):
        return len(self.df_xy) * 512

    def __getitem__(self, index):
        if self.is_test:
            return self._epochs(index)
        else:
            return self._epochs_amp(index)

    def _epochs_amp(self, index):
        """
        随机调整每个channel的'幅值'与'极性'
        :param index:
        :return:
        """

        fif_idx = index % len(self.df_xy)
        label = self.labels[fif_idx]
        fif_data = self.dataset_dict[fif_idx]   # torch.tensor() @
        n_times = fif_data.shape[-1]

        sel_offset = np.random.randint(n_times-self.samples, size=2)
        x1 = fif_data[:, :, sel_offset[0]: sel_offset[0]+self.samples]
        x2 = fif_data[:, :, sel_offset[1]: sel_offset[1]+self.samples]

        amp, noise = self._amp()

        x5 = x1 * amp[0] + noise[0]
        x6 = x2 * amp[1] + noise[1]

        return (x5, x6), label

    def _amp(self):
        # 以q的概率衰减幅值
        # q = 0.25
        n = torch.rand((2, 1, 30, 1))
        a = n * 0.25 + 0.75     # [0.75, 1.0)
        a[n > self.q] = 1.0

        # 以p的概率翻转极性
        # p = 0.25
        n = torch.rand((2, 1, 30, 1))
        s = torch.ones((2, 1, 30, 1))
        s[n < self.p] = -1.0

        # noise = torch.randn((2, 1, 30, 1)) * 1.0e-4
        noise = torch.randn((2, 1, 30, self.samples)) * self.r

        return a * s, noise

    def _epochs(self, index):
        fif_idx = index % len(self.df_xy)
        label = self.labels[fif_idx]
        fif_data = self.dataset_dict[fif_idx]   # torch.tensor() @
        n_times = fif_data.shape[-1]

        sel_offset = np.random.randint(n_times-self.samples, size=2)
        x1 = fif_data[:, :, sel_offset[0]: sel_offset[0]+self.samples]
        x2 = fif_data[:, :, sel_offset[1]: sel_offset[1]+self.samples]

        return (x1, x2), label


class DatasetSimSiamAmp4(Dataset):
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
        if self.is_test:
            return self._epoch_end(index)
        else:
            return self._epoch_amp(index)

    def _epoch_amp(self, index):
        """
        相邻的2个epochs
        随机调整每个channel的'幅值'与'极性'
        :param index:
        :return:
        """

        fif_idx = index % len(self.df_xy)
        label = self.labels[fif_idx]
        fif_data = self.dataset_dict[fif_idx]   # torch.tensor() @
        n_times = fif_data.shape[-1]

        # 相邻的2个epochs
        sel_offset = np.random.randint(n_times-self.samples*2)
        x1 = fif_data[:, :, sel_offset: sel_offset + self.samples]

        sel_offset += self.samples
        x2 = fif_data[:, :, sel_offset: sel_offset + self.samples]

        amp, noise = self._amp()

        x5 = x1 * amp[0] + noise[0]
        x6 = x2 * amp[1] + noise[1]

        return (x5, x6), label

    def _amp(self):
        # 以q的概率衰减幅值
        q = 0.25
        n = torch.rand((2, 1, 30, 1))
        a = n * 0.25 + 0.75     # [0.75, 1.0)
        a[n > q] = 1.0

        # 以p的概率翻转极性
        p = 0.25
        n = torch.rand((2, 1, 30, 1))
        s = torch.ones((2, 1, 30, 1))
        s[n < p] = -1.0

        noise = torch.randn((2, 1, 30, 1)) * 1.0e-4

        return a * s, noise

    def _epoch_end(self, index):
        fif_idx = index % len(self.df_xy)
        label = self.labels[fif_idx]
        fif_data = self.dataset_dict[fif_idx]   # torch.tensor() @
        n_times = fif_data.shape[-1]

        sel_offset = np.random.randint(n_times//2, n_times-self.samples, size=2)
        x1 = fif_data[:, :, sel_offset[0]: sel_offset[0]+self.samples]
        x2 = fif_data[:, :, sel_offset[1]: sel_offset[1]+self.samples]

        return (x1, x2), label


class DatasetSimSiamAmp4b(Dataset):
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
        if self.is_test:
            return self._epoch_end(index)
        else:
            return self._epoch_amp(index)

    def _epoch_amp(self, index):
        """
        相邻的2个epochs
        随机调整每个channel的'幅值'与'极性'
        :param index:
        :return:
        """

        fif_idx = index % len(self.df_xy)
        label = self.labels[fif_idx]
        fif_data = self.dataset_dict[fif_idx]   # torch.tensor() @
        n_times = fif_data.shape[-1]

        # 相隔一个 self.samples 的2个 epochs
        sel_offset = np.random.randint(n_times-self.samples*3)
        x1 = fif_data[:, :, sel_offset: sel_offset + self.samples]

        sel_offset += self.samples * 2
        x2 = fif_data[:, :, sel_offset: sel_offset + self.samples]

        amp, noise = self._amp()

        x5 = x1 * amp[0] + noise[0]
        x6 = x2 * amp[1] + noise[1]

        return (x5, x6), label

    def _amp(self):
        # 以q的概率衰减幅值
        q = 0.25
        n = torch.rand((2, 1, 30, 1))
        a = n * 0.25 + 0.75     # [0.75, 1.0)
        a[n > q] = 1.0

        # 以p的概率翻转极性
        p = 0.25
        n = torch.rand((2, 1, 30, 1))
        s = torch.ones((2, 1, 30, 1))
        s[n < p] = -1.0

        noise = torch.randn((2, 1, 30, 1)) * 1.0e-4

        return a * s, noise

    def _epoch_end(self, index):
        fif_idx = index % len(self.df_xy)
        label = self.labels[fif_idx]
        fif_data = self.dataset_dict[fif_idx]   # torch.tensor() @
        n_times = fif_data.shape[-1]

        sel_offset = np.random.randint(n_times//2, n_times-self.samples, size=2)
        x1 = fif_data[:, :, sel_offset[0]: sel_offset[0]+self.samples]
        x2 = fif_data[:, :, sel_offset[1]: sel_offset[1]+self.samples]

        return (x1, x2), label


class DatasetSimSiamAmp5(Dataset):
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
        if self.is_test:
            return self._epoch_end(index)
        else:
            return self._epoch_amp(index)

    def _epoch_amp(self, index):
        """
        相邻的2个epochs
        随机调整每个channel的'幅值'与'极性'
        :param index:
        :return:
        """

        fif_idx = index % len(self.df_xy)
        label = self.labels[fif_idx]
        fif_data = self.dataset_dict[fif_idx]   # torch.tensor() @
        n_times = fif_data.shape[-1]

        # 相邻[0, self.samples)的2个epochs
        sel_offset = np.random.randint(n_times-self.samples*3)
        x1 = fif_data[:, :, sel_offset: sel_offset + self.samples]

        sel_offset = np.random.randint(sel_offset + self.samples, sel_offset + self.samples*2)
        x2 = fif_data[:, :, sel_offset: sel_offset + self.samples]

        amp, noise = self._amp()

        x5 = x1 * amp[0] + noise[0]
        x6 = x2 * amp[1] + noise[1]

        return (x5, x6), label

    def _amp(self):
        # 以q的概率衰减幅值
        q = 0.25
        n = torch.rand((2, 1, 30, 1))
        a = n * 0.25 + 0.75     # [0.75, 1.0)
        a[n > q] = 1.0

        # 以p的概率翻转极性
        p = 0.25
        n = torch.rand((2, 1, 30, 1))
        s = torch.ones((2, 1, 30, 1))
        s[n < p] = -1.0

        noise = torch.randn((2, 1, 30, 1)) * 1.0e-5

        return a * s, noise

    def _epoch_end(self, index):
        fif_idx = index % len(self.df_xy)
        label = self.labels[fif_idx]
        fif_data = self.dataset_dict[fif_idx]   # torch.tensor() @
        n_times = fif_data.shape[-1]

        sel_offset = np.random.randint(n_times//2, n_times-self.samples, size=2)
        x1 = fif_data[:, :, sel_offset[0]: sel_offset[0]+self.samples]
        x2 = fif_data[:, :, sel_offset[1]: sel_offset[1]+self.samples]

        return (x1, x2), label


class DatasetSimSiamAmp6(Dataset):
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
        if self.is_test:
            return self._epoch_end(index)
        else:
            return self._epoch_amp(index)

    def _epoch_amp(self, index):
        """
        noise = torch.randn((2, 1, 30, 1)) * 1.0e-4
        相邻的2个epochs
        随机调整每个channel的'幅值'与'极性'
        :param index:
        :return:
        """

        fif_idx = index % len(self.df_xy)
        label = self.labels[fif_idx]
        fif_data = self.dataset_dict[fif_idx]   # torch.tensor() @
        n_times = fif_data.shape[-1]

        # 相邻的2个epochs
        sel_offset = np.random.randint(n_times-self.samples*3)
        # sel_offset = np.random.randint(n_times//2, n_times-self.samples, size=2)
        x1 = fif_data[:, :, sel_offset: sel_offset + self.samples]
        # sel_offset += self.samples
        sel_offset = np.random.randint(sel_offset + self.samples, sel_offset + self.samples*2)
        x2 = fif_data[:, :, sel_offset: sel_offset + self.samples]

        amp, noise = self._amp()

        x5 = x1 * amp[0] + noise[0]
        x6 = x2 * amp[1] + noise[1]

        return (x5, x6), label

    def _amp(self):
        # 以q的概率衰减幅值
        q = 0.25
        n = torch.rand((2, 1, 30, 1))
        a = n * 0.25 + 0.75     # [0.75, 1.0)
        a[n > q] = 1.0

        # 以p的概率翻转极性
        p = 0.25
        n = torch.rand((2, 1, 30, 1))
        s = torch.ones((2, 1, 30, 1))
        s[n < p] = -1.0

        noise = torch.randn((2, 1, 30, 1)) * 1.0e-4

        return a * s, noise

    def _epoch_end(self, index):
        fif_idx = index % len(self.df_xy)
        label = self.labels[fif_idx]
        fif_data = self.dataset_dict[fif_idx]   # torch.tensor() @
        n_times = fif_data.shape[-1]

        sel_offset = np.random.randint(n_times//2, n_times-self.samples, size=2)
        x1 = fif_data[:, :, sel_offset[0]: sel_offset[0]+self.samples]
        x2 = fif_data[:, :, sel_offset[1]: sel_offset[1]+self.samples]

        return (x1, x2), label


class DatasetSimSiamAmp7(Dataset):
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
        if self.is_test:
            return self._epoch_end(index)
        else:
            return self._epoch_amp(index)

    def _epoch_amp(self, index):
        """
        noise = torch.randn((2, 1, 30, 1)) * 1.0e-4
        相邻的2个epochs
        随机调整每个channel的'幅值'与'极性'
        :param index:
        :return:
        """

        fif_idx = index % len(self.df_xy)
        label = self.labels[fif_idx]
        fif_data = self.dataset_dict[fif_idx]   # torch.tensor() @
        n_times = fif_data.shape[-1]

        # 相邻的2个epochs
        sel_offset = np.random.randint(n_times-self.samples*3)
        # sel_offset = np.random.randint(n_times//2, n_times-self.samples, size=2)
        x1 = fif_data[:, :, sel_offset: sel_offset + self.samples]
        # sel_offset += self.samples
        # sel_offset = np.random.randint(sel_offset + self.samples, sel_offset + self.samples*2)
        sel_offset = np.random.randint(sel_offset, sel_offset + self.samples*2)
        x2 = fif_data[:, :, sel_offset: sel_offset + self.samples]

        amp, noise = self._amp()

        x5 = x1 * amp[0] + noise[0]
        x6 = x2 * amp[1] + noise[1]

        return (x5, x6), label

    def _amp(self):
        # 以q的概率衰减幅值
        q = 0.25
        n = torch.rand((2, 1, 30, 1))
        a = n * 0.25 + 0.75     # [0.75, 1.0)
        a[n > q] = 1.0

        # 以p的概率翻转极性
        p = 0.25
        n = torch.rand((2, 1, 30, 1))
        s = torch.ones((2, 1, 30, 1))
        s[n < p] = -1.0

        noise = torch.randn((2, 1, 30, 1)) * 1.0e-4

        return a * s, noise

    def _epoch_end(self, index):
        fif_idx = index % len(self.df_xy)
        label = self.labels[fif_idx]
        fif_data = self.dataset_dict[fif_idx]   # torch.tensor() @
        n_times = fif_data.shape[-1]

        sel_offset = np.random.randint(n_times//2, n_times-self.samples, size=2)
        x1 = fif_data[:, :, sel_offset[0]: sel_offset[0]+self.samples]
        x2 = fif_data[:, :, sel_offset[1]: sel_offset[1]+self.samples]

        return (x1, x2), label


class DatasetSimSiamAmp8(Dataset):
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
        if self.is_test:
            return self._epoch_end(index)
        else:
            return self._epoch_amp(index)

    def _epoch_amp(self, index):
        """
        noise = torch.randn((2, 1, 30, 1)) * 1.0e-4
        相邻的2个epochs
        随机调整每个channel的'幅值'与'极性'
        :param index:
        :return:
        """

        fif_idx = index % len(self.df_xy)
        label = self.labels[fif_idx]
        fif_data = self.dataset_dict[fif_idx]   # torch.tensor() @
        n_times = fif_data.shape[-1]

        # 前后两部分的2个epochs
        sel_offset = np.random.randint(n_times//2)
        x1 = fif_data[:, :, sel_offset: sel_offset + self.samples]

        sel_offset = np.random.randint(n_times//2, n_times-self.samples)
        x2 = fif_data[:, :, sel_offset: sel_offset + self.samples]

        amp, noise = self._amp()

        x5 = x1 * amp[0] + noise[0]
        x6 = x2 * amp[1] + noise[1]

        return (x5, x6), label

    def _amp(self):
        # 以q的概率衰减幅值
        q = 0.25
        n = torch.rand((2, 1, 30, 1))
        a = n * 0.25 + 0.75     # [0.75, 1.0)
        a[n > q] = 1.0

        # 以p的概率翻转极性
        p = 0.25
        n = torch.rand((2, 1, 30, 1))
        s = torch.ones((2, 1, 30, 1))
        s[n < p] = -1.0

        noise = torch.randn((2, 1, 30, 1)) * 1.0e-5

        return a * s, noise

    def _epoch_end(self, index):
        fif_idx = index % len(self.df_xy)
        label = self.labels[fif_idx]
        fif_data = self.dataset_dict[fif_idx]   # torch.tensor() @
        n_times = fif_data.shape[-1]

        sel_offset = np.random.randint(n_times//2, n_times-self.samples, size=2)
        x1 = fif_data[:, :, sel_offset[0]: sel_offset[0]+self.samples]
        x2 = fif_data[:, :, sel_offset[1]: sel_offset[1]+self.samples]

        return (x1, x2), label


class DatasetSimSiamAmp9(Dataset):
    def __init__(self, x, y=None, dataset_dict=None,
                 samples=251, stride=250, channels=1,
                 is_test=False):
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
        self.stride = stride

    def __len__(self):
        return len(self.df_xy) * 512

    def __getitem__(self, index):
        if self.is_test:
            return self._epoch_end(index)
        else:
            return self._epoch_amp(index)

    def _epoch_amp(self, index):
        """
        noise = torch.randn((2, 1, 30, 1)) * 1.0e-4
        相邻的2个epochs
        随机调整每个channel的'幅值'与'极性'
        :param index:
        :return:
        """

        fif_idx = index % len(self.df_xy)
        label = self.labels[fif_idx]
        fif_data = self.dataset_dict[fif_idx]   # torch.tensor() @
        n_times = fif_data.shape[-1]

        # 间隔一个stride的2个epochs, 有重叠
        sel_offset = np.random.randint(n_times//2)
        x1 = fif_data[:, :, sel_offset: sel_offset + self.samples]

        sel_offset += self.stride
        x2 = fif_data[:, :, sel_offset: sel_offset + self.samples]

        amp, noise = self._amp()

        x5 = x1 * amp[0] + noise[0]
        x6 = x2 * amp[1] + noise[1]

        return (x5, x6), label

    def _amp(self):
        # 以q的概率衰减幅值
        q = 0.25
        n = torch.rand((2, 1, 30, 1))
        a = n * 0.25 + 0.75     # [0.75, 1.0)
        a[n > q] = 1.0

        # 以p的概率翻转极性
        p = 0.25
        n = torch.rand((2, 1, 30, 1))
        s = torch.ones((2, 1, 30, 1))
        s[n < p] = -1.0

        noise = torch.randn((2, 1, 30, 1)) * 1.0e-5

        return a * s, noise

    def _epoch_end(self, index):
        fif_idx = index % len(self.df_xy)
        label = self.labels[fif_idx]
        fif_data = self.dataset_dict[fif_idx]   # torch.tensor() @
        n_times = fif_data.shape[-1]

        sel_offset = np.random.randint(n_times//2, n_times-self.samples, size=2)
        x1 = fif_data[:, :, sel_offset[0]: sel_offset[0]+self.samples]
        x2 = fif_data[:, :, sel_offset[1]: sel_offset[1]+self.samples]

        return (x1, x2), label


class DatasetSimSiamDouble(Dataset):
    def __init__(self, x, y=None, dataset_dict=None, duration=251, channels=2):
        """
        :param x: Dataframe
        :param y: 可忽略
        """
        self.x = x
        self.dataset_dict = dataset_dict
        self.duration = duration
        self.channels = channels

    def __len__(self):
        return len(self.x) * 512

    def __getitem__(self, index):
        fif_idx = index % len(self.x)
        # file = self.x.loc[fif_idx]['file']
        label = self.x.loc[fif_idx]['label']

        fif_data = self.dataset_dict[fif_idx]
        n_times = fif_data.shape[-1]
        # print(fif_data.shape)   # (1, 30, n_times)

        sel_offset = np.random.randint(self.duration//2, size=self.channels)
        x1 = fif_data[:, :, sel_offset[0]: sel_offset[0]+self.duration]
        x2 = fif_data[:, :, sel_offset[1]: sel_offset[1]+self.duration]

        sel_offset = np.random.randint(self.duration//2, n_times-self.duration, size=self.channels)
        x3 = fif_data[:, :, sel_offset[0]: sel_offset[0]+self.duration]
        x4 = fif_data[:, :, sel_offset[1]: sel_offset[1]+self.duration]

        x1 = np.concatenate([x1, x3], axis=-2)
        x2 = np.concatenate([x2, x4], axis=-2)

        x1 = torch.tensor(x1)
        x2 = torch.tensor(x2)
        label = torch.tensor(label)

        return (x1, x2), label


class DatasetSimSiamDouble200(Dataset):
    def __init__(self, x, y=None, dataset_dict=None, samples=251, channels=2):
        """
        :param x: Dataframe
        :param y: 可忽略
        """
        self.x = x
        self.dataset_dict = dataset_dict
        self.samples = samples
        self.channels = channels
        self.labels = torch.tensor(self.x['label'].to_numpy())

    def __len__(self):
        return len(self.x) * 512

    def __getitem__(self, index):
        fif_idx = index % len(self.x)
        # file = self.x.loc[fif_idx]['file']
        # label = self.x.loc[fif_idx]['label']
        label = self.labels[fif_idx]

        fif_data = self.dataset_dict[fif_idx]
        n_times = fif_data.shape[-1]
        # print(fif_data.shape)   # (1, 30, n_times)

        # sel_offset = np.random.randint(self.duration//2, size=self.channels)
        sel_offset = np.random.randint(n_times-self.samples, size=self.channels * 2)
        x1 = fif_data[:, :, sel_offset[0]: sel_offset[0]+self.samples]
        x2 = fif_data[:, :, sel_offset[1]: sel_offset[1]+self.samples]
        x3 = fif_data[:, :, sel_offset[2]: sel_offset[2]+self.samples]
        x4 = fif_data[:, :, sel_offset[3]: sel_offset[3]+self.samples]

        # x1 = np.concatenate([x1, x3], axis=-2)
        # x2 = np.concatenate([x2, x4], axis=-2)

        x1 = torch.concat([x1, x3], dim=-2)
        x2 = torch.concat([x2, x4], dim=-2)

        # x1 = torch.tensor(x1)
        # x2 = torch.tensor(x2)

        # x1 = torch.from_numpy(x1)
        # x2 = torch.from_numpy(x2)
        # label = torch.tensor(label)

        return (x1, x2), label


class DatasetDF(Dataset):
    """
    继承 DataSequence, 数据集采用 DF+dict 方式.
    x: DataFrame
    y: None
    dataset_dict: dict
    """

    def __init__(self, x, y=None, sample_weight=None, dataset_dict=None,
                 label_noise=0.0, axis=1, method='sn', one_hot=False, samples=251):
        """
        :param x: np.array or Dataframe
        :param y: 如果x是Dataframe类型, 可忽略
        :param sample_weight: 样本权重 None|np.array
        :param label_noise: DisturbLabel rate [0.0, 0.2]
        :param axis: 合并多个epochs成example时指定的轴, 用于扩展example的高度
        :param method: 合并多个epochs之方法
        :param one_hot: bool, 是否需要one_hot编码
        """

        self.df_xy = x
        self.dataset_dict = dataset_dict
        self.samples = samples
        self.sample_weight = sample_weight
        self.axis = axis
        self.one_hot = one_hot
        self.table_one_hot = np.array([[1.0, 0.0], [0.0, 1.0]], dtype='float32')

        self.files_path = self.df_xy['file'].to_numpy()
        self.sn_str = self.df_xy['sn_str'].to_numpy()
        self.labels = self.df_xy['label'].to_numpy()

    def __len__(self):
        return len(self.df_xy)

    def __getitem__(self, index):
        x, y = self._example_by_sn_dict(index)

        # if self.sample_weight is not None:
        #     w = self.sample_weight[y]
        #     w = torch.tensor(w)
        #
        # if self.one_hot:
        #     y = self.table_one_hot[y]

        # x = torch.tensor(x).to(torch.float32)
        x = torch.from_numpy(x).to(torch.float32)
        y = torch.tensor(y).to(torch.float32)

        # # 样本权重, 用于evaluate时的样本均衡. 作用可忽略
        # if self.sample_weight is not None:
        #     return (x, y), w
        # else:
        #     return x, y

        return x, y

    def _example_by_shuffle_dict(self, idx):
        file_path = self.df_xy['file'].iloc[idx]
        sn_str_of_example = self.df_xy['sn_str'].iloc[idx]  # 多个epochs文件的后缀序号

        x_all = self.dataset_dict[file_path]

        # 依次加载epochs, 置乱后合并
        xn_list = []
        for sn in sn_str_of_example.split(','):
            xn_epoch = x_all[:, :, int(sn): int(sn) + self.samples, :]
            xn_list.append(xn_epoch)

        np.random.shuffle(xn_list)

        if len(xn_list) > 1:
            example = np.concatenate(xn_list, axis=self.axis)
        else:
            example = xn_list[0]

        return example

    def _example_by_sn_dict(self, idx):
        # file_path = self.df_xy['file'].iloc[idx]
        file_path = self.files_path[idx]
        x_all = self.dataset_dict[file_path]

        # sn_str_of_example = self.df_xy['sn_str'].iloc[idx]  # 多个epochs文件的后缀序号
        sn_str_of_example = self.sn_str[idx]  # 多个epochs文件的后缀序号

        # 依次加载后合并
        xn_list = []
        for sn in sn_str_of_example.split(','):
            xn_epoch = x_all[:, :, int(sn): int(sn) + self.samples]
            xn_list.append(xn_epoch)

        if len(xn_list) > 1:
            example = np.concatenate(xn_list, axis=self.axis)
        else:
            example = xn_list[0]

        # label = self.df_xy['label'].iloc[idx]
        label = self.labels[idx]
        # print(label)

        return example, label

    def load_data_with_df_dict_offset(self, indices_):
        """
        DataFrame索引数据的加载.
        df包含多个epochs文件的序号后缀, 需加载后按指定轴合并.
        :param indices_:
        :return:
        """
        x_batch = []
        y_batch = []
        for idx in indices_:
            if self.method == 'shuffle':
                xn = self._example_by_shuffle_dict(idx)
            else:
                xn = self._example_by_sn_dict(idx)

            x_batch.append(xn)

            yn = self.df_xy['label'].iloc[idx]
            y_batch.append(yn)

        x_batch = np.concatenate(x_batch, axis=0)
        y_batch = np.array(y_batch)

        # 人为加扰label(DisturbLabel)
        if self.label_noise > 0.0:
            y_batch = self._disturb_label(y_batch)

        if self.sample_weight is not None:
            w = self.sample_weight[y_batch]

        if self.one_hot:
            y_batch = self.table_one_hot[y_batch]

        # 样本权重, 用于evaluate时的样本均衡. 作用可忽略
        if self.sample_weight is not None:
            return x_batch, y_batch, w
        else:
            return x_batch, y_batch

