# mne imports
import mne
#from mne import io

# EEGNet-specific imports

# tools for plotting confusion matrices

# PyRiemann imports
import re

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import torch.nn.functional as F

from cfg6 import *
import logging

# from SimSiamLib.EEGLab_Model import *
from SimSiamLib.LearningRate import *
from SimSiamLib.EEGLab_Tools import *
from SimSiamLib.SimSiam_Logging import *
from SimSiamLib.SimSiam_Dataset import DatasetSimSiamCouple
from SimSiamLib.SimSiam_Models import EEGNet_PT452E_TC16_V3, SimSiam

from SimSiamLib.Constraints import MaxNorm, UnitNorm
from SimSiamLib.EEGLab_Transform import *


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def _load_ch_time(file_path: str, preload=True):
    """
    支援 raw 與 -epo.fif。
    回傳 (data[ch, time], info, n_times_total, is_epochs, epochs_3d or None)
    """
    if file_path.endswith('-epo.fif') or file_path.endswith('-epo.fif.gz'):
        epochs = mne.read_epochs(file_path, preload=True, verbose=False)
        X = epochs.get_data()                       # (n_ep, ch, T)
        data = np.concatenate(list(X), axis=-1)     # (ch, n_ep*T)
        return data, epochs.info, data.shape[-1], True, X
    else:
        raw = mne.io.read_raw_fif(file_path, preload=preload, verbose=False)
        return raw.get_data(), raw.info, raw.n_times, False, None


class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name; self.fmt = fmt
        self.val = 0; self.avg = 0; self.sum = 0; self.count = 0
    def update(self, val, n=1):
        self.val = val; self.sum += val * n; self.count += n; self.avg = self.sum / self.count
    def get_value(self): return self.avg
    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class EEGLabSimSiam(object):
    def __init__(self):
        self.duration = eeg_config['duration']
        self.stride = eeg_config['stride']
        self.overlap = self.duration - self.stride
        self.channel_factor = eeg_config['CF']
        self.samples = int(self.duration * 250.) + 1
        self.eeg_config = eeg_config

        self.noise_rate = 0.2
        self.split_random_state = 0
        self.sn_kfold = 0

        self.class_weight = {0: 1.0, 1: 1.0}
        self.sample_weight = {0: 1.0, 1: 1.0}

        self.eeg_channels = ['FP1','FP2','F7','F3','FZ','F4','F8',
                             'FT7','FC3','FCZ','FC4','FT8',
                             'T3','C3','CZ','C4','T4',
                             'TP7','CP3','CPZ','CP4','TP8',
                             'T5','P3','PZ','P4','T6',
                             'O1','OZ','O2']
        self.eeg_channel_num = len(self.eeg_channels)

        self.n_classes = len(class_names)
        self.class_names = class_names

        self.code_file = os.path.split(__file__)[-1]
        self.code_file_name, ext = os.path.splitext(self.code_file)
        class_all = '-'.join(class_names)
        self.bone_file_name = '_'.join([self.code_file_name, class_all]) + '_' + '-'.join(keyword)

        self.info_file   = self.bone_file_name + '_info.csv'
        self.info_all_file = self.bone_file_name + '_info_all.csv'
        self.epochs_file = self.bone_file_name + '_epochs.csv'
        self.examples_file = self.bone_file_name + '_examples.csv'
        self.state_dict_file = self.bone_file_name + '.pt'

        self.df_dataset_info = None
        self.df_epochs_info = None
        self.dataset_dict = None

        self.log = SimSiamLogging(user=self)

    # ------------ 統一的讀檔函式：raw 或 epochs 都可 ------------
    def _load_ch_time(self, file_path, preload=True):
        """
        回傳 (data[ch, time], info, n_times_total, is_epochs, epochs_3d or None)
        """
        if file_path.endswith('-epo.fif') or file_path.endswith('-epo.fif.gz'):
            epochs = mne.read_epochs(file_path, preload=True, verbose=False)
            X = epochs.get_data()  # (n_ep, ch, T)
            data = np.concatenate(list(X), axis=-1)  # (ch, n_ep*T)
            return data, epochs.info, data.shape[-1], True, X
        else:
            raw = mne.io.read_raw_fif(file_path, preload=preload, verbose=False)
            return raw.get_data(), raw.info, raw.n_times, False, None

    def register_logging(self):
        logging.basicConfig(filename=self.bone_file_name + '.log', level=logging.INFO,
                            format='%(asctime)s %(message)s')
        for key in eeg_config:
            val = eeg_config[key]
            if isinstance(val, float): logging.info(key + ': %f' % (val,))
            if isinstance(val, bool):  logging.info(key + ': %s' % (val,))
            elif isinstance(val, int): logging.info(key + ': %d' % (val,))
            if isinstance(val, str):   logging.info(key + ': %s' % (val,))
            if isinstance(val, type):  logging.info(key + ': %s' % (str(val), ))
        logging.info('\n')

    def collect_dataset_info_all(self):
        df_dict = {'file': [], 'eeg_name': [], 'EC-EO': [], 'label': [], 'n_times': [], 'std': [], 'std2': [], 'std3': []}
        for eeg_name in eeg_class_names:
            for root, _, files in os.walk(os.path.join(base_path, eeg_class_paths[eeg_name])):
                for file in files:
                    if not (file.endswith('.fif') or file.endswith('.fif.gz')):
                        continue
                    file_path = os.path.join(root, file)

                    data, info, n_total, is_epo, _ = self._load_ch_time(file_path, preload=False)
                    if len(info['ch_names']) < self.eeg_channel_num:
                        continue

                    xn = data
                    df_dict['std'].append(np.sqrt(np.mean(np.square(xn))))
                    c_std = np.std(xn, axis=1); df_dict['std2'].append(c_std.mean())
                    df_dict['std3'].append(np.std(xn))

                    print(file_path)
                    df_dict['n_times'].append(n_total)
                    df_dict['EC-EO'].append(file[:2])  # 只保留前兩碼（你為 'DC'）
                    df_dict['file'].append(file_path)
                    df_dict['eeg_name'].append(eeg_name)
                    df_dict['label'].append(eeg_class_labels[eeg_name])

        pd.DataFrame(df_dict).to_csv(self.info_all_file)

    def collect_dataset_info(self):
        df_dict = {'file': [], 'eeg_name': [], 'EC-EO': [], 'label': [], 'n_times': [], 'std': [], 'std2': [], 'std3': []}
        for eeg_name in eeg_class_names:
            for root, _, files in os.walk(os.path.join(base_path, eeg_class_paths[eeg_name])):
                for file in files:
                    if not (file.endswith('.fif') or file.endswith('.fif.gz')):
                        continue
                    # keyword 過濾（只看前兩碼）
                    if file[:2] not in keyword:
                        continue

                    file_path = os.path.join(root, file)
                    data, info, n_total, is_epo, _ = self._load_ch_time(file_path, preload=False)
                    if len(info['ch_names']) < self.eeg_channel_num:
                        continue

                    xn = data
                    df_dict['std'].append(np.sqrt(np.mean(np.square(xn))))
                    c_std = np.std(xn, axis=1); df_dict['std2'].append(c_std.mean())
                    df_dict['std3'].append(np.std(xn))

                    print(file_path)
                    df_dict['n_times'].append(n_total)
                    df_dict['EC-EO'].append(file[:2])
                    df_dict['file'].append(file_path)
                    df_dict['eeg_name'].append(eeg_name)
                    df_dict['label'].append(eeg_class_labels[eeg_name])

        pd.DataFrame(df_dict).to_csv(self.info_file)

    def load_fif_data_to_mem(self, normalize=True):
        print('>>', 'load_fif_data_to_mem()')
        mkdir(epochs_path)
        df_dataset_info = pd.read_csv(self.info_file)
        self.dataset_dict = {}
        for idx in tqdm(range(len(df_dataset_info))):
            file_path = df_dataset_info.iloc[idx]['file'].strip()
            data, info, _, is_epo, _ = self._load_ch_time(file_path, preload=True)
            raw_data = data  # (ch, n_times)
            if normalize:
                c_std = np.std(raw_data, axis=1, keepdims=True); xn_std = c_std.mean()
                x_all = (raw_data / (xn_std + 1e-8))[None, :, :].astype('float32')
            else:
                x_all = (raw_data[None, :, :] * 1000).astype('float32')
            self.dataset_dict[idx] = torch.from_numpy(x_all)
        total = sum(self.dataset_dict[k].shape[2] for k in self.dataset_dict)
        print('\r', total, sep='')
        return self.dataset_dict

    def load_fif_data_to_mem_all(self, normalize=True):
        print('>>', 'load_fif_data_to_mem_all()')
        mkdir(epochs_path)
        df_dataset_info_all = pd.read_csv(self.info_all_file)
        self.dataset_dict = {}
        for idx in tqdm(range(len(df_dataset_info_all))):
            file_path = df_dataset_info_all.iloc[idx]['file'].strip()
            data, info, _, is_epo, _ = self._load_ch_time(file_path, preload=True)
            raw_data = data
            if normalize:
                c_std = np.std(raw_data, axis=1, keepdims=True); xn_std = c_std.mean()
                x_all = (raw_data / (xn_std + 1e-8))[None, :, :].astype('float32')
            else:
                x_all = (raw_data[None, :, :] * 1000).astype('float32')
            self.dataset_dict[idx] = torch.from_numpy(x_all)
        total = sum(self.dataset_dict[k].shape[2] for k in self.dataset_dict)
        print('\r', total, sep='')
        return self.dataset_dict

    def fif_file_to_epochs_array_from_disk(self, fif_file, coef):
        """
        若為 -epo.fif：直接回傳該檔案內的 epochs。
        若為 raw .fif：用 duration/stride 切窗。
        回傳形狀： (N, 1, ch, samples)
        """
        if fif_file.endswith('-epo.fif') or fif_file.endswith('-epo.fif.gz'):
            epochs = mne.read_epochs(fif_file.strip(), preload=True, verbose=False)
            X = epochs.get_data()  # (N, ch, T_epoch)
            if eeg_config['data_normalize']:
                c_std = np.std(X, axis=2, keepdims=True)
                X = X / (c_std.mean(axis=1, keepdims=True) + 1e-8) * 1.0e-2
            X = X.astype('float32')
            return X[:, None, :, :]

        raw = mne.io.read_raw_fif(fif_file.strip(), preload=True, verbose=False)
        n_times = raw.n_times
        raw_data = raw.get_data(verbose=False)  # (ch, n_times)

        if eeg_config['data_normalize']:
            c_std = np.std(raw_data, axis=1, keepdims=True); xn_std = c_std.mean()
            x_all = (raw_data / (xn_std + 1e-8))[None, :, :].astype('float32')
        else:
            x_all = (raw_data[None, :, :] * 1000).astype('float32')

        stride = int(self.stride * 250.0 / max(coef, 1e-8))
        n_epochs = (n_times - self.samples) // stride + 1
        offsets = np.arange(n_epochs) * stride

        epochs_all = [x_all[:, :, off: off + self.samples] for off in offsets]
        return np.array(epochs_all)  # (N, 1, ch, samples)


    def __init__(self):
        self.duration = eeg_config['duration']
        self.stride = eeg_config['stride']
        self.overlap = self.duration - self.stride
        self.channel_factor = eeg_config['CF']

        self.samples = int(self.duration * 250.) + 1
        self.eeg_config = eeg_config

        self.noise_rate = 0.2  # DisturbLabel

        self.split_random_state = 0             # 随机分割数据集的种子, 被log收集. user.sn_kfold
        self.sn_kfold = 0                       # KFold的序号, 被log收集. user.sn_kfold

        self.class_weight = {0: 1.0, 1: 1.0}    # 训练样本均衡补偿, 加载数据经统计后更新.
        self.sample_weight = {0: 1.0, 1: 1.0}   # 验证样本均衡补偿, 加载数据经统计后更新.

        self.eeg_channels = ['FP1', 'FP2',
                             'F7', 'F3', 'FZ', 'F4', 'F8',
                             'FT7', 'FC3', 'FCZ', 'FC4', 'FT8',
                             'T3', 'C3', 'CZ', 'C4', 'T4',
                             'TP7', 'CP3', 'CPZ', 'CP4', 'TP8',
                             'T5', 'P3', 'PZ', 'P4', 'T6',
                             'O1', 'OZ', 'O2']

        self.eeg_channel_num = len(self.eeg_channels)
        self.n_classes = len(class_names)
        self.class_names = class_names

        # 自动获取代码文件名, 生成系列文件名的主体
        self.code_file = os.path.split(__file__)[-1]  # 获取代码文件名 *.py
        self.code_file_name, ext = os.path.splitext(self.code_file)  #
        class_all = '-'.join(class_names)
        self.bone_file_name = '_'.join([self.code_file_name, class_all]) + '_' + '-'.join(keyword)

        # info文件名
        self.info_file = self.bone_file_name + '_info.csv'
        self.info_all_file = self.bone_file_name + '_info_all.csv'
        self.epochs_file = self.bone_file_name + '_epochs.csv'
        self.examples_file = self.bone_file_name + '_examples.csv'

        self.state_dict_file = self.bone_file_name + '.pt'

        # dataFrame数据结构
        self.df_dataset_info = None
        self.df_epochs_info = None

        self.dataset_dict = None

        # 实例化EEGLabLogging类, 模型与历史文件管理
        self.log = SimSiamLogging(user=self)

    ######################################################

    def register_logging(self):
        logging.basicConfig(filename=self.bone_file_name + '.log', level=logging.INFO,
                            format='%(asctime)s %(message)s')

        for key in eeg_config:
            val = eeg_config[key]
            if isinstance(val, float):
                logging.info(key + ': %f' % (val,))

            if isinstance(val, bool):
                logging.info(key + ': %s' % (val,))
            elif isinstance(val, int):
                logging.info(key + ': %d' % (val,))

            if isinstance(val, str):
                logging.info(key + ': %s' % (val,))

            if isinstance(val, type):
                logging.info(key + ': %s' % (str(val), ))

        logging.info('\n')

    ######################################################
    @staticmethod
    def _first_int_or(index_like, default_val):
        """把 pandas Index 取第一個整數；若空則回傳 default_val。"""
        try:
            return int(index_like[0])
        except Exception:
            return int(default_val)

    @staticmethod
    def _trial_stem(file_name):
        """
        從檔名中移除 trial 編號，取得同 subject 的共同前綴。
        例如：DCD1_xxx_pre_trial002-epo.fif -> DCD1_xxx_pre_
        找不到 'trial' 就回傳前 27~30 字元的前綴（維持你原本行為的精神）。
        """
        m = re.search(r'(.*?)(?:_trial\d+)(.*)$', file_name)
        if m:
            return m.group(1)  # '..._pre'
        # fallback：保留前綴，避免過度匹配
        return file_name[: max(2, min(30, len(file_name)))]


    def append_couple_index(self):
        df_info_all = pd.read_csv(self.info_all_file, index_col=0)
        df_info = pd.read_csv(self.info_file)

        # 先建立欄位
        df_info['couple_idx'] = df_info.index
        df_info['self_idx'] = 0

        for index in df_info.index:
            file_name_path = str(df_info.loc[index, 'file']).strip()
            base = os.path.split(file_name_path)[-1]

            # 找到在「全集」中對應的那一列（用 endswith 可避免路徑差異）
            df_self = df_info_all[df_info_all['file'].str.endswith(base, na=False)]
            self_idx = self._first_int_or(df_self.index, index)
            df_info.at[index, 'self_idx'] = self_idx

            # 同 subject 的「伴侶」：拿掉 _trial### 後的共同前綴當 key
            stem = self._trial_stem(base)
            # 先排除自己，再找包含相同 stem 的其他列
            df_others = df_info_all[df_info_all.index != self_idx]
            df_couple = df_others[df_others['file'].str.contains(re.escape(stem), regex=True, na=False)]

            # 若找到多個，就取第一個；找不到就回退成自己
            couple_idx = self._first_int_or(df_couple.index, self_idx)
            df_info.at[index, 'couple_idx'] = couple_idx

        df_info.to_csv(self.info_file)
        df_info_all.to_csv(self.info_all_file)

    def append_couple_index_all(self):
        df_info_all = pd.read_csv(self.info_all_file, index_col=0)

        df_info_all['couple_idx'] = df_info_all.index
        df_info_all['self_idx'] = df_info_all.index

        for index in df_info_all.index:
            file_name_path = str(df_info_all.loc[index, 'file']).strip()
            base = os.path.split(file_name_path)[-1]

            # 同 subject 的伴侶（在全集內找）
            stem = self._trial_stem(base)
            df_others = df_info_all[df_info_all.index != index]
            df_couple = df_others[df_others['file'].str.contains(re.escape(stem), regex=True, na=False)]

            couple_idx = self._first_int_or(df_couple.index, index)
            df_info_all.at[index, 'couple_idx'] = couple_idx

        df_info_all.to_csv(self.info_all_file)

    def collect_dataset_info_all(self):
        """
        加载' All' 数据集文件, 基础数据目录存盘为 *_info_all.csv.
        :return: None
        """

        # 用于生成DataFrame(字典生成方法)
        df_dict = {'file': [], 'eeg_name': [], 'EC-EO': [], 'label': [], 'n_times': [], 'std': [], 'std2': [], 'std3': []}

        # 收集数据集, df_dict
        for eeg_name in eeg_class_names:
            files_list = []
            for root, _, files in os.walk(os.path.join(base_path, eeg_class_paths[eeg_name])):
                for file in files:
                    file_path = os.path.join(root, file)

                    data, info, n_total, is_epo, _ = _load_ch_time(file_path, preload=False)
                    if len(info['ch_names']) < self.eeg_channel_num:
                        continue

                    # # EC | EO
                    # if not file.startswith(keyword):
                    #     continue
                    # if file[:2] not in keyword:
                    #     continue

                    xn = data
                    df_dict['std'].append(np.sqrt(np.mean(np.square(xn))))
                    c_std = np.std(xn, axis=1); df_dict['std2'].append(c_std.mean())
                    df_dict['std3'].append(np.std(xn))

                    # file_path = file_path.ljust(75)  # 补空格, 目的在于csv文件的可读性.
                    # files_list.append(file_path)
                    print(file_path)

                    df_dict['n_times'].append(n_total)
                    df_dict['EC-EO'].append(file[:2])
                    df_dict['file'].append(file_path)
                    df_dict['eeg_name'].append(eeg_name)
                    df_dict['label'].append(eeg_class_labels[eeg_name])

        # 生成 DataFrame
        df_dataset_info = pd.DataFrame(df_dict, index=[n for n in range(len(df_dict['file']))])

        df_dataset_info.to_csv(self.info_all_file)

    def collect_dataset_info(self):
        """
        加载数据集文件, 基础数据目录存盘为 *_info.csv.
        :return: None
        """

        # 用于生成DataFrame(字典生成方法)
        df_dict = {'file': [], 'eeg_name': [], 'EC-EO': [], 'label': [], 'n_times': [], 'std': [], 'std2': [], 'std3': []}

        # 收集数据集, df_dict
        for eeg_name in eeg_class_names:
            files_list = []
            for root, _, files in os.walk(os.path.join(base_path, eeg_class_paths[eeg_name])):
                for file in files:
                    file_path = os.path.join(root, file)

                    if not (file.endswith('.fif') or file.endswith('.fif.gz')):
                        continue
                    if file[:2] not in keyword:   # 你若不想過濾可刪這行
                        continue

                    data, info, n_total, is_epo, _ = _load_ch_time(file_path, preload=False)
                    if len(info['ch_names']) < self.eeg_channel_num:
                        continue

                    # # EC | EO
                    # if not file.startswith(keyword):
                    #     continue
                    if file[:2] not in keyword:
                        continue

                    xn = data
                    df_dict['std'].append(np.sqrt(np.mean(np.square(xn))))
                    c_std = np.std(xn, axis=1); df_dict['std2'].append(c_std.mean())
                    df_dict['std3'].append(np.std(xn))

                    # file_path = file_path.ljust(75)  # 补空格, 目的在于csv文件的可读性.
                    # files_list.append(file_path)
                    print(file_path)

                    df_dict['n_times'].append(n_total)
                    df_dict['EC-EO'].append(file[:2])
                    df_dict['file'].append(file_path)
                    df_dict['eeg_name'].append(eeg_name)
                    df_dict['label'].append(eeg_class_labels[eeg_name])

        df_dataset_info = pd.DataFrame(df_dict, index=[n for n in range(len(df_dict['file']))])

        df_dataset_info.to_csv(self.info_file)

    ######################################################

    def _normalize(self, xn):
        # xn_std = np.sqrt(np.mean(np.square(xn)))
        # df_dict['std'].append(xn_std)

        c_std = np.std(xn, axis=1)
        xn_std = c_std.mean()
        # df_dict['std2'].append(xn_std)
        #
        # xn_std = np.std(xn)
        # df_dict['std3'].append(xn_std)

        return xn / xn_std * 1.0e-2  # 1.0e-5 * 1000

    def load_fif_data_to_mem(self, normalize=True):
        """
        分割fif成epochs, 依次编号存入字典.
        :return:
        """
        import sys
        from tqdm import tqdm

        print('>>', 'load_fif_data_to_mem()')

        epochs_disk_path = epochs_path
        mkdir(epochs_disk_path)

        df_dataset_info = pd.read_csv(self.info_file)
        self.dataset_dict = {}

        # 逐行处理df_dataset_info
        for idx in tqdm(range(len(df_dataset_info))):
            row = df_dataset_info.iloc[idx]  # 读一行
            file_path = row['file'].strip()

            data, info, _, _, _ = _load_ch_time(file_path, preload=True)  # (ch, n_times)
            raw_data = data
            if normalize:
                c_std = np.std(raw_data, axis=1, keepdims=True); xn_std = c_std.mean()
                x_all = (raw_data / (xn_std + 1e-8))[None, :, :].astype('float32')
            else:
                x_all = (raw_data[None, :, :] * 1000).astype('float32')
            self.dataset_dict[idx] = torch.from_numpy(x_all)

        total = 0
        for key in self.dataset_dict:
            total += self.dataset_dict[key].shape[2]
        print('\r', total, sep='')

        return self.dataset_dict

    def load_fif_data_to_mem_all(self, normalize=True):
        """
        分割fif成epochs, 依次编号存入字典.
        :return:
        """
        import sys
        from tqdm import tqdm

        print('>>', 'load_fif_data_to_mem_all()')

        epochs_disk_path = epochs_path
        mkdir(epochs_disk_path)

        df_dataset_info_all = pd.read_csv(self.info_all_file)
        self.dataset_dict = {}

        # 逐行处理df_dataset_info
        for idx in tqdm(range(len(df_dataset_info_all))):
            row = df_dataset_info_all.iloc[idx]  # 读一行
            file_path = row['file'].strip()

            data, info, _, _, _ = _load_ch_time(file_path, preload=True)  # (ch, n_times)
            raw_data = data
            if normalize:
                c_std = np.std(raw_data, axis=1, keepdims=True); xn_std = c_std.mean()
                x_all = (raw_data / (xn_std + 1e-8))[None, :, :].astype('float32')
            else:
                x_all = (raw_data[None, :, :] * 1000).astype('float32')
            self.dataset_dict[idx] = torch.from_numpy(x_all)

            self.dataset_dict[idx] = torch.from_numpy(x_all)   # for torch

        total = 0
        for key in self.dataset_dict:
            total += self.dataset_dict[key].shape[2]
        print('\r', total, sep='')

        return self.dataset_dict

    ########################################################

    def build_train_loader(self, batch_size=64):
        """
        :return:
        """

        df_dataset_info_all = pd.read_csv(self.info_all_file)
        df_dataset_info = pd.read_csv(self.info_file)
        dict_data = self.load_fif_data_to_mem_all(normalize=eeg_config['data_normalize'])

        dataset = eeg_config['dataset']

        train_dataset = dataset(df_dataset_info_all,        # df_dataset_info_all,    #
                                dataset_dict=dict_data,
                                samples=self.samples,
                                is_test=False,
                                channels=self.channel_factor)

        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=0,
                                                   pin_memory=True,
                                                   sampler=None,
                                                   drop_last=True)

        return train_loader

    def train_torch(self, epochs=200, batch_size=256, accum_iter=4):
        """

        :param epochs:
        :param batch_size:
        :param accum_iter: 累积梯度
        :return:
        """
        import time

        # select model
        backbone_net = eeg_config['back_bone']
        backbone = backbone_net(nb_classes=eeg_config['dim'],
                                kern_length=eeg_config['kern_length'],
                                fc_dim=eeg_config['dim'])

        model = SimSiam(backbone, eeg_config['dim'], eeg_config['pred_dim'])
        print(model)
        model.cuda(device)

        # define loss function (criterion) and optimizer
        criterion = nn.CosineSimilarity(dim=1).cuda(device)
        optimizer = torch.optim.SGD(model.parameters(),
                                    eeg_config['lr_init'],
                                    momentum=0.9,
                                    weight_decay=eeg_config['weight_decay'])

        # 自动生成模型文件名/训练历史文件名
        model_file, history_file = self.log.gen_model_history_file_name_from_log_file(model.__class__.__name__)

        # Monitor: 自定义的监测记录class
        monitor = SimSiamMonitor()

        train_loader = self.build_train_loader(batch_size=batch_size)

        # 逐回合训练
        val_acc_best = 0.0
        loss_best = 1.0e6
        for epoch in range(epochs):
            print("\nEpoch {0}/{1}".format(str(epoch + 1), str(epochs)))

            # 0. set learning rate every epoch
            lr = adjust_learning_rate(optimizer, eeg_config['lr_init'], epoch, epochs)

            t0 = time.time()

            # batch accumulation parameter
            accum_iter = accum_iter
            losses_meter = AverageMeter('Loss', ':.4f')

            model.train()

            # 1. 对每一个batch进行训练
            for batch_idx, (images, _) in enumerate(train_loader):
                images[0] = images[0].cuda(device, non_blocking=True)
                images[1] = images[1].cuda(device, non_blocking=True)

                # compute output and loss
                p1, p2, z1, z2 = model(x1=images[0], x2=images[1])
                loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5

                # 用于计算1个回合内的平均值
                losses_meter.update(loss.item(), images[0].size(0))

                # # compute gradient and do SGD step
                # optimizer.zero_grad()
                # loss.backward()
                # optimizer.step()

                # 梯度累加
                # https://zhuanlan.zhihu.com/p/595716023

                # scale the loss to the mean of the accumulated batch size
                loss = loss / accum_iter

                # backward pass
                loss.backward()

                # weights update
                if ((batch_idx + 1) % accum_iter == 0) or (batch_idx + 1 == len(train_loader)):
                    optimizer.step()
                    optimizer.zero_grad()

                    # max_norm constrains
                    # print(model.state_dict().keys())
                    # print(model._modules['encoder']._modules['spatial_conv'])
                    # model._modules['encoder']['spatial_conv'][0].apply(MaxNorm(1.0, -2))
                    # w0 = model._modules['encoder']._modules['spatial_conv'][0].weight.data.flatten()

                    model._modules['encoder']._modules['spatial_conv'][0].apply(MaxNorm(1.0, -2))

                    # w1 = model._modules['encoder']._modules['spatial_conv'][0].weight.data.flatten()
                    # print(w1 - w0)

            # print(model.state_dict()['encoder.temporal_conv.0.weight'].shape)
            # print(model.encoder.temporal_conv[0].weight.shape)
            # print(model.encoder.spatial_conv[0].weight.shape)

            print(losses_meter)
            print(time.time() - t0)
            if losses_meter.get_value() < loss_best:
                loss_best = losses_meter.get_value()
                torch.save(model.state_dict(), model_file)

            monitor.update(losses_meter.get_value(), lr)

            # 5. save the latest model for transform every epoch
            self.log.save_final_model_history(model, pd.DataFrame(monitor.history))

    ########################################################

    def build_predict_model(self, log_index=-1):
        # select model
        # model = SimSiam(EEGNet_V452_Valid_Residual(nb_classes=eeg_config['dim'], samples=self.samples, channels=self.channel_factor),
        #                 eeg_config['dim'], eeg_config['pred_dim'])

        backbone_net = eeg_config['back_bone']
        backbone = backbone_net(nb_classes=eeg_config['dim'],
                                kern_length=eeg_config['kern_length'],
                                fc_dim=eeg_config['dim']
                                )

        model = SimSiam(backbone, eeg_config['dim'], eeg_config['pred_dim'])

        df_log = pd.read_csv(self.log.log_file)
        state_dict_file = df_log['pt_csv'].iloc[log_index] + '.pt'

        state_dict = torch.load(state_dict_file)
        model.load_state_dict(state_dict, strict=False)

        features_model = model.encoder

        # 冻结梯度
        for param in model.parameters():
            param.requires_grad = False

        # 替换fc层
        features_model.fc = nn.Identity()

        return features_model

    def build_predict_model2(self, log_index=-1):
        # select model
        back_bone = eeg_config['back_bone']
        backbone = back_bone(nb_classes=eeg_config['dim'],
                             samples=self.samples,
                             channels=self.channel_factor)

        model = SimSiam(backbone, eeg_config['dim'], eeg_config['pred_dim'])

        df_log = pd.read_csv(self.log.log_file)
        state_dict_file = df_log['pt_csv'].iloc[log_index] + '.pt'

        state_dict = torch.load(state_dict_file)
        model.load_state_dict(state_dict, strict=False)

        features_model = model.encoder

        # 冻结梯度
        for param in model.parameters():
            param.requires_grad = False

        # # 替换fc层
        # features_model.fc = nn.Identity()
        features_model.fc[-1] = nn.Identity()

        return features_model

    def predict_torch(self, log_index=-1):
        """

        :return:
        """

        print('>>', 'predict_torch(log_index=%d)' % (log_index, ))

        # select model
        model = self.build_predict_model(log_index=log_index)
        # model = self.build_feature_model2(log_index=log_index)
        print(model)
        model.cuda(device)

        # train_loader = self.build_train_loader(batch_size=batch_size)

        features_disk_path = features_path
        mkdir(features_disk_path)

        df_info = pd.read_csv(self.info_file)
        df_dataset_info = pd.read_csv(self.info_all_file)

        if eeg_config['over_sampling']:
            over_sampling_coef = self.over_sampling(df_info)
        else:
            over_sampling_coef = [1., 1.]

        # 逐行处理df_dataset_info
        for index in tqdm(df_dataset_info.index):
            row = df_dataset_info.loc[index]  # 读一行
            file_path = row['file'].strip()
            label = row['label']

            epochs_array = self.fif_file_to_epochs_array_from_disk(file_path, over_sampling_coef[label])
            epochs_tensor = torch.from_numpy(epochs_array)

            model.eval()
            with torch.no_grad():
                x = epochs_tensor.cuda(device, non_blocking=True)
                features = model(x)
                features = features.cpu()

                # f_n = F.normalize(features, dim=1)
                # f_s = f_n.mm(f_n.T)
                # f_w = torch.softmax(f_s, dim=1)
                # features = f_w.mm(features)
                # print(f_n.shape, f_s.shape, f_w.shape, features.shape)

                features = features.numpy()

            # 存盘
            base_file = os.path.split(file_path)[-1]
            features_file_path = os.path.join(features_disk_path, base_file + '.pkl')

            with open(features_file_path, 'wb') as f:
                pickle.dump(features, f)

    def over_sampling(self, df):
        """
        计算不同类别examples的不均衡度, 不同的方法对acc会有一定影响.
        :param df:
        :return: class_weight dict
        """
        over_sampling_coef = {0: 1.0, 1: 1.0}
        df_0 = df[df['label'] == 0]
        df_1 = df[df['label'] == 1]
        if len(df_1) > len(df_0):
            over_sampling_coef[0] = len(df_1) / len(df_0)
        else:
            over_sampling_coef[1] = len(df_0) / len(df_1)

        print(over_sampling_coef)

        # over_sampling_coef = {0: 1.0, 1: 1.0}
        # # df = df_dataset_info
        # df_0_n_times = df[df['label'] == 0]['n_times'].sum()
        # df_1_n_times = df[df['label'] == 1]['n_times'].sum()
        # if df_1_n_times > df_0_n_times:
        #     over_sampling_coef[0] = df_1_n_times / df_0_n_times
        # else:
        #     over_sampling_coef[1] = df_0_n_times / df_1_n_times
        # print(over_sampling_coef)

        return over_sampling_coef

    def fif_file_to_epochs_array_from_disk(self, fif_file, coef):
        """
        在固定分割的epochs中, 随机采样多个epochs,
        合成单一example, 增加 channels. 用于evaluate
        :param fif_file: str
        :param coef: float
        :return:
        """
        # 先處理 -epo.fif：直接回傳現成 epochs
        if fif_file.endswith('-epo.fif') or fif_file.endswith('-epo.fif.gz'):
            epochs = mne.read_epochs(fif_file.strip(), preload=True, verbose=False)
            X = epochs.get_data()  # (N, ch, T_epoch)
            if eeg_config['data_normalize']:
                c_std = np.std(X, axis=2, keepdims=True)
                X = X / (c_std.mean(axis=1, keepdims=True) + 1e-8) * 1.0e-2
            return X.astype('float32')[:, None, :, :]  # (N, 1, ch, T)

        # raw .fif：切窗
        raw = mne.io.read_raw_fif(fif_file.strip(), preload=True, verbose=False)
        raw_data = raw.get_data(verbose=False)  # (ch, n_times)
        if eeg_config['data_normalize']:
            c_std = np.std(raw_data, axis=1, keepdims=True); xn_std = c_std.mean()
            x_all = (raw_data / (xn_std + 1e-8))[None, :, :].astype('float32')
        else:
            x_all = (raw_data[None, :, :] * 1000).astype('float32')

        stride = int(self.stride * 250.0 / max(coef, 1e-8))
        n_epochs = (raw.n_times - self.samples) // stride + 1
        offsets = np.arange(n_epochs) * stride
        epochs_all = [x_all[:, :, off: off + self.samples] for off in offsets]
        return np.array(epochs_all)  # (N, 1, ch, samples)

    ########################################################

    def t_SNE(self):
        """
        t-SNE降维, 可视化feature
        :return:
        """
        from tqdm import tqdm

        df_dataset_info = pd.read_csv(self.info_file)

        # EC only
        df_dataset_info = df_dataset_info[df_dataset_info['EC-EO'] == 'EC']
        # df_dataset_info = df_dataset_info[df_dataset_info['label'] == 0]
        # df_dataset_info = df_dataset_info[df_dataset_info['label'] == 1]

        labels = df_dataset_info['label'].to_numpy()

        feature_list = []
        for index in tqdm(df_dataset_info.index, colour='green'):
            row = df_dataset_info.loc[index]
            file_path = row['file']

            base_file = os.path.split(file_path)[-1]
            features_file_path = os.path.join(features_path, base_file + '.pkl')
            with open(features_file_path, 'rb') as f:
                features = pickle.load(f)

            # features = SelfAttention(T=0.5)(features)

            # # normalize features
            # features = features / np.linalg.norm(features, axis=1, keepdims=True)

            # 特征均值
            Z = np.mean(features, axis=0, keepdims=True)

            # # normalize
            # Z = Z / np.linalg.norm(Z)

            feature_list.append(Z)
        feature = np.concatenate(feature_list, axis=0)

        self._t_sne(feature, labels, name='Z_All')

        plt.show()

    def _t_sne(self, feature, labels, name='2D'):
        from sklearn.manifold import TSNE

        fig_path = os.path.join('./Fig', self.code_file_name)
        mkdir(fig_path)

        # tsne = TSNE(n_components=2, learning_rate=50,
        #             # method='exact',
        #             init='random', verbose=2)

        tsne = TSNE(n_components=2, verbose=2,
                    learning_rate=200,
                    n_iter=2000, n_iter_without_progress=500)

        feature_tsne = tsne.fit_transform(feature)
        print(feature_tsne.shape)

        plt.figure(name)

        # # plt.scatter(feature_tsne[:, 0], feature_tsne[:, 1], c=pos)
        # plt.scatter(feature_tsne[pos == 0, 0], feature_tsne[pos == 0, 1], c='r')
        # plt.scatter(feature_tsne[pos == 1, 0], feature_tsne[pos == 1, 1], c='g')
        # plt.scatter(feature_tsne[pos == 2, 0], feature_tsne[pos == 2, 1], c='b')

        # 逐label绘制散点图
        cm = ['k', 'r', 'b', 'y', 'g']
        for n in range(max(labels)+1):
            plt.scatter(feature_tsne[labels == n, 0], feature_tsne[labels == n, 1],
                        c=cm[n],
                        s=5,
                        label=n)

        # for i in range(feature_tsne.shape[0]):
        #     plt.text(feature_tsne[i, 0], feature_tsne[i, 1], str(i))

        plt.legend()

        fig_name = name + '.png'
        plt.savefig(os.path.join(fig_path, fig_name))

        # plt.show()

    def t_SNE_epochs(self, g=2):
        """
        t-SNE降维, 可视化每个 fif 之 epochs 的 feature
        :return:
        """
        from tqdm import tqdm

        df_dataset_info = pd.read_csv(self.info_file)
        df_dataset_info = df_dataset_info[df_dataset_info['EC-EO'] == 'EC']

        for index in tqdm(df_dataset_info.index, colour='green'):
            row = df_dataset_info.loc[index]
            file_path = row['file']

            base_file = os.path.split(file_path)[-1]
            features_file_path = os.path.join(features_path, base_file + '.pkl')
            with open(features_file_path, 'rb') as f:
                feature = pickle.load(f)

                # normalize
                feature = feature / np.linalg.norm(feature, axis=1, keepdims=True)

                # 按时间先后分组
                g_list = np.array_split(np.arange(feature.shape[0]), g)

                flags_list = []
                for g_array, flag in zip(g_list, range(g)):
                    g_array[:] = flag
                    print(g_array)
                    flags_list.append(g_array)
                flags = np.concatenate(flags_list)

                self._t_sne(feature, flags, name='%s_%s' % (g, index))

    def t_SNE_some_epochs(self):
        """
            t-SNE降维, 可视化feature
        :return:
        """
        from tqdm import tqdm

        df_dataset_info = pd.read_csv(self.info_file)
        # EC only
        df_dataset_info = df_dataset_info[df_dataset_info['EC-EO'] == 'EC']

        labels = []
        feature_list = []
        for index in tqdm(df_dataset_info.index):
            # if index % 2 != 0:
            #     continue

            row = df_dataset_info.loc[index]
            file_path = row['file']
            label = row['label']

            base_file = os.path.split(file_path)[-1]
            features_file_path = os.path.join(features_path, base_file + '.pkl')
            with open(features_file_path, 'rb') as f:
                features = pickle.load(f)

            features = SelfAttention(T=0.5)(features)

            feature_list.append(features)
            labels += [label] * features.shape[0]

        feature = np.concatenate(feature_list, axis=0)
        labels = np.array(labels)

        self._t_sne(feature, labels, name='Epochs')

        plt.show()


def main_train():
    eeg = EEGLabSimSiam()

    eeg.train_torch(epochs=eeg_config['epochs'],
                    batch_size=eeg_config['batch_size'],
                    accum_iter=eeg_config['accum_iter'])


# 添加配置文件
eeg_config = {'duration': 1.5,
              'stride': 1.0,
              'CF': 1,
              'kern_length': 125,

              'data_normalize': False,  # fif数据是否标准化

              'lr_init': 5.0e-2 * 4,
              'weight_decay': 1.0e-4,

              'epochs': 20,
              'batch_size': 256,
              'accum_iter': 4,

              'back_bone': EEGNet_PT452E_TC16_V3,
              'dataset': DatasetSimSiamCouple,    # DatasetSimSiamCoupleBalance,

              # dim: feature dimension (default: 2048)
              # pred_dim: hidden dimension of the predictor (default: 512)
              # 'dim': 2048,
              # 'pred_dim': 512

              'dim': 512,
              'pred_dim': 128,

              'over_sampling': False,

              'Note': 'Contrastive learning with EC/EO from one subject.'
              }


# keyword = ['EC', 'EO']
keyword = ['DC']


py_code_file = os.path.split(__file__)[-1]  # 获取代码文件名 *.py
py_file_name, _ = os.path.splitext(py_code_file)
features_path = './Features/' + '%s_%s_%s' % (py_file_name, str(eeg_config['duration']), str(eeg_config['stride']))


if __name__ == '__main__':
    eeg = EEGLabSimSiam()
    # print(features_path)

    def init():
        eeg.collect_dataset_info()
        eeg.collect_dataset_info_all()

        eeg.append_couple_index()
        eeg.append_couple_index_all()
        # eeg.load_fif_data_to_mem()

    # step 1
    init()
    main_train()

    # step 2
    eeg.predict_torch(log_index=-1)

    # # step 3
    # eeg.t_SNE()
    # eeg.t_SNE_some_epochs()

    #
    # eeg.t_SNE_epochs()

    # os.system("shutdown -s -t 240")

    def balabala():
        eeg.build_train_loader()
        eeg.train_torch()



