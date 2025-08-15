# mne imports
import mne
import pandas as pd
from mne import io

# EEGNet-specific imports

# tools for plotting confusion matrices
from matplotlib import pyplot as plt

# PyRiemann imports

import os
import json

from EEGLab.EEGLab_Transform import *

# from utils.EEGLab_Model import *
from utils.LearningRate import *
from utils.EEGLab_Tools import *
from utils.EEGLab_Log import EEGLabLogging

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

import logging

from cfg6 import *

from EEGLab.utils import AccuracyMeter, AverageMeter

from Co_teachingLab.Models import EEGNet_PT452_FC1
from Co_teachingLab.Loss import *
from Co_teachingLab.Dataset import DatasetCoTeachingArray, DatasetCoTeachingArrayBalance
from Co_teachingLab.Co_teaching_Logging import CoTeachingLogging

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# from EEGLab_Testing_V452III_Base import *


def limit_gpu_memory(memory_limit=512):
    """
    以下部分代码, 用于限制GPU内存增长. 可以封装成函数, 但必须在main中首先调用, 才能生效!
    使用 tf.config.experimental.set_virtual_device_configuration 配置虚拟 GPU 设备，
    并且设置可在 GPU 上分配多少总内存的硬性限制。
    https://www.tensorflow.org/guide/gpu?hl=zh-cn
    :return:
    """
    import tensorflow as tf

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only allocate memory_limit(MB) of memory on the first GPU
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)])

            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)
    else:
        print('no GPU', gpus)


class StratifiedKFoldDF(object):
    def __init__(self, n_splits=5, shuffle=True, random_state=0):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, df):
        from sklearn.model_selection import StratifiedKFold

        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)
        self.random_state += 1000

        for train_pos, test_pos in skf.split(df.index.to_numpy(), df['label'].to_numpy()):
            df_train = df.loc[df.index[train_pos]]
            df_test = df.loc[df.index[test_pos]]
            yield df_train, df_test


class EEGLabLOSOFeature(object):
    """
    Co-teaching！
    """

    def __init__(self):
        super().__init__()

        self.duration = eeg_config['duration']
        self.stride = eeg_config['stride']
        self.overlap = self.duration - self.stride
        self.channel_factor = eeg_config['CF']

        self.eeg_config = eeg_config

        self.class_weight = {0: 1.0, 1: 1.0}  # 训练样本均衡补偿, 加载数据经统计后更新.
        self.sample_weight = {0: 1.0, 1: 1.0}  # 验证样本均衡补偿, 加载数据经统计后更新.

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
        code_file = os.path.split(__file__)[-1]  # 获取代码文件名 *.py
        code_file_name, ext = os.path.splitext(code_file)  #
        class_all = '-'.join(class_names)
        kw_tag = keyword if isinstance(keyword, str) else '+'.join(keyword)
        self.bone_file_name = '_'.join([code_file_name, class_all]) + '_' + kw_tag
        
        # info文件名
        self.info_file = self.bone_file_name + '_info.csv'
        self.epochs_file = self.bone_file_name + '_epochs.csv'

        self.evaluate_file = self.bone_file_name + '_evaluate.csv'

        # dataFrame数据结构
        self.df_dataset_info = None
        self.df_epochs_info = None

        # 实例化 CoTeachingLogging 类, 模型与历史文件管理
        self.log = CoTeachingLogging(user=self)

        # 声明. 避免后续赋值时出现的"下划波浪线"
        self.x_train, self.x_val, self.x_test = None, None, None
        self.y_train, self.y_val, self.y_test = None, None, None

        self.df_train, self.df_val, self.df_test = None, None, None

        self.dataset_dict = None
        self.dataset_array = None
        self.dataset_labels = None

        self.fit_method = None

        self.group = 0

    def n_epochs_to_sn_group(self, n_epochs, coef):
        n_epochs = int(n_epochs)
        epochs_sel = np.arange(n_epochs)
        epochs_sel = epochs_sel.reshape(-1, 1)
        epochs_sel = np.repeat(epochs_sel, self.channel_factor, axis=1)
        return epochs_sel

    def register_logging(self):
        logging.basicConfig(filename=self.bone_file_name + '.log', level=logging.INFO,
                            format='%(asctime)s %(message)s')

        # logging.info('duration: %.2f' % (self.duration,))
        # logging.info('stride: %.2f' % (self.stride,))
        # logging.info('channel_factor: %d' % (self.channel_factor,))
        #
        # logging.info('samples_balance: %s' % (eeg_config['samples_balance'],))
        # # logging.info('test_set_shuffle: %s' % (eeg_config['test_set_shuffle'], ))
        #
        # # logging.info('noise_rate: %.2f' % (eeg_config['noise_rate'], ))
        # logging.info('lr_init: %.2e' % (eeg_config['lr_init'],))

        for key in eeg_config:
            val = eeg_config[key]
            if isinstance(val, float):
                logging.info(key + ': %f' % (val,))

            if isinstance(val, bool):
                logging.info(key + ': %s' % (val,))

            if isinstance(val, int):
                logging.info(key + ': %d' % (val,))

            if isinstance(val, str):
                logging.info(key + ': %s' % (val,))

        logging.info('\n')

    ######################################################

    def make_epochs_sn_df(self):
        """

        :return:
        """
        print('>>', 'make_epochs_sn_df()')
        mkdir(epochs_path)

        df_dataset_info = pd.read_csv(self.info_file)

        # 计算train样本不均衡率
        over_sampling_coef = {0: 1.0, 1: 1.0}

        if eeg_config['samples_balance']:
            df = df_dataset_info
            df_0 = df[df['label'] == 0]
            df_1 = df[df['label'] == 1]
            if len(df_1) > len(df_0):
                over_sampling_coef[0] = len(df_1) / len(df_0)
            else:
                over_sampling_coef[1] = len(df_0) / len(df_1)
        print(over_sampling_coef)

        df_dict = {'file': [], 'upper_limit': [], 'fif_idx': [], 'ID': [], 'sn_str': [], 'label': [], 'Black_List': []}

        # 逐行处理df_dataset_info
        for idx in range(len(df_dataset_info)):
            row = df_dataset_info.iloc[idx]  # 读一行
            file_path = row['file'].strip()
            label = row['label']
            n_times = int(row['n_times'])

            if file_path.endswith('-epo.fif') or file_path.endswith('-epo.fif.gz'):
                # 這裡的 n_times 其實是「epoch 數」
                sn_sel = self.n_epochs_to_sn_group(n_times, over_sampling_coef[label])
            else:
                # raw 檔才用時間窗切片
                sn_sel = self.n_times_to_epochs_sn_group(n_times, over_sampling_coef[label])

            # 以fif文件名基础, 生成的epochs独立编排序号
            base_file = os.path.split(file_path)[-1]

            # (组成example的) epochs序号存入df
            for m in range(sn_sel.shape[0]):
                example_sn_str = [str(idx).zfill(4) for idx in sn_sel[m]]

                sn_str = ','.join(example_sn_str)
                df_dict['sn_str'].append(sn_str)

                epoch_file = base_file + '_xy'
                epoch_path = os.path.join(epochs_path, epoch_file)
                df_dict['file'].append(epoch_path)

            df_dict['label'] += [label] * sn_sel.shape[0]
            df_dict['upper_limit'] += [sn_sel.shape[0]] * sn_sel.shape[0]
            df_dict['ID'] += list(range(sn_sel.shape[0]))
            df_dict['fif_idx'] += [idx] * sn_sel.shape[0]

            try:
                df_dict['Black_List'] += [row['Black_List']] * sn_sel.shape[0]
            except Exception as e:
                df_dict['Black_List'] += [False] * sn_sel.shape[0]
                print(e)

        df_examples_info = pd.DataFrame(df_dict, index=[n for n in range(len(df_dict['file']))])
        df_examples_info.to_csv(self.epochs_file)

    def n_times_to_epochs_sn_group(self, n_times, coef):
        """
        按设置参数分割fif至epochs group.
        :param n_times: int
        :param coef: float
        :return: x_all
        """
        stride = int(self.stride * 250. / coef)
        duration = int(self.duration * 250.0 + 1)
        n_epochs = (n_times - duration) // stride + 1

        epochs_sel = np.arange(n_epochs)
        epochs_sel = epochs_sel.reshape(-1, 1)
        epochs_sel = np.repeat(epochs_sel, self.channel_factor, axis=1)

        return epochs_sel

    ######################################################

    def load_fif_data_to_mem(self):
        """
        不分割fif成epochs, 将raw_data整体存入字典.
        :return:
        """
        print('>>', 'load_fif_data_to_mem()')

        epochs_disk_path = epochs_path

        df_dataset_info = pd.read_csv(self.info_file)
        self.dataset_dict = {}

        # 逐行处理df_dataset_info
        for idx in df_dataset_info.index:
            row = df_dataset_info.iloc[idx]  # 读一行
            file_path = row['file'].strip()
            print(file_path)

            # 以fif文件名基础, 生成内存字典的key
            base_file = os.path.split(file_path)[-1]
            epoch_file = base_file + '_xy'
            epoch_file_path = os.path.join(epochs_disk_path, epoch_file)

            raw = io.Raw(file_path, preload=True, verbose=False)
            raw_data = raw.get_data(verbose=False)  # (30, n_times)

            x_all = raw_data[None, :, :, None] * 1000  # (1, 30, n_times, 1)
            x_all = x_all.astype('float32')

            self.dataset_dict[epoch_file_path] = x_all

        total = 0
        for key in self.dataset_dict:
            total += self.dataset_dict[key].shape[2]
        print(total)

    def load_feature_data_to_mem(self, normalize=False):
        """
        不分割fif成epochs, 将raw_data整体存入字典.
        :return:
        """
        print('>>', 'load_feature_data_to_mem()')

        df_dataset_info = pd.read_csv(self.info_file)
        self.dataset_dict = {}

        # 逐行处理df_dataset_info
        for idx in df_dataset_info.index:
            row = df_dataset_info.iloc[idx]  # 读一行
            file_path = row['file'].strip()

            base_file = os.path.split(file_path)[-1]
            features_file_path = os.path.join(features_path, base_file + '.pkl')
            print(features_file_path)

            with open(features_file_path, 'rb') as f:
                x_all = pickle.load(f)

                if eeg_config['data_transform']:
                    x_all = transformer_like(x_all)
                    # x_all = transformer_like_scaling(x_all)

            if eeg_config['data_normalize']:
                x_all = x_all / np.linalg.norm(x_all, axis=1, keepdims=True)

            self.dataset_dict[idx] = torch.from_numpy(x_all)#x_all

    def load_feature_data_to_array(self):
        """
        從 features_path 掃出所有 .pkl 檔，透過「規範化後的檔名 key」對應 info.csv，
        讀到就加入樣本，沒對到就跳過並列出清單。
        """
        import re
        from pathlib import Path


        def norm_key_strip(name_or_path: str) -> str:
            """
            把路徑/檔名正規化成可比對的 key：
            - 兼容 / 與 \（用 Path 取檔名）
            - 去掉副檔名 .fif/.fif.gz/.pkl
            - 去掉尾端 -epo/_epo/-raw 及可能跟著的其他段
            - 去掉 `_pre_trial###` 或 `-trial###` 這類 trial 編號
            - 統一大小寫與 -/_ 連字
            """
            base = Path(str(name_or_path)).name  # 處理 / 與 \ 混用
            # 去副檔名
            base = re.sub(r'\.fif(\.gz)?$', '', base, flags=re.I)
            base = re.sub(r'\.pkl$', '', base, flags=re.I)

            # 去尾碼（epo/raw + 任何後續片段）
            base = re.sub(r'[-_](epo|raw)(?:[-_].*)?$', '', base, flags=re.I)

            # 去 trial 後綴：_pre_trial001、-trial002、_trial2... 都砍掉
            base = re.sub(r'([-_]pre)?[-_]?trial\d+$', '', base, flags=re.I)

            # 也有人會把 trial 放在中間：..._pre_trial001_clean 這種
            base = re.sub(r'([-_]pre)?[-_]?trial\d+([-_].*)?$', '', base, flags=re.I)

            # 視需要移除類別前綴（若 .pkl 有 CTL_/CM_ 而 info 沒有）
            # base = re.sub(r'^(ctl|cm)[-_]+', '', base, flags=re.I)

            base = base.strip()
            base = re.sub(r'\s+', '', base)
            base = re.sub(r'[-_]+', '_', base)  # 把 -/_ 視為相同分隔
            return base.lower()

        def norm_info_key(path: str) -> str:
            return norm_key_strip(path)

        def norm_pkl_key(pkl_path: str) -> str:
            return norm_key_strip(pkl_path)

        print('>> load_feature_data_to_array()')

        if not os.path.isdir(features_path):
            raise FileNotFoundError(f"features_path 不存在：{features_path}")

        # 1) 建立 features 目錄的索引：key -> pkl 路徑
        pkl_index = {}
        for root, _, files in os.walk(features_path):
            for f in files:
                if f.lower().endswith('.pkl'):
                    p = os.path.join(root, f)
                    pkl_index[norm_pkl_key(p)] = p
        print(f"[scan] 在 features_path 找到 {len(pkl_index)} 個 .pkl")

        # 2) 讀 info.csv，用 key 對應到 pkl_index
        df = pd.read_csv(self.info_file)
        x_list, l_list, miss = [], [], []
        for _, row in df.iterrows():
            key = norm_info_key(str(row['file']).strip())
            pkl_path = pkl_index.get(key)
            if pkl_path is None:
                miss.append(key)
                continue

            with open(pkl_path, 'rb') as f:
                x_all = pickle.load(f)

            # 可選轉換/正規化
            if eeg_config.get('data_transform', False):
                T = eeg_config.get('T', 1.0)
                x_all = SelfAttention(T=T)(x_all)
            if eeg_config.get('data_normalize', False):
                denom = np.linalg.norm(x_all, axis=1, keepdims=True)
                denom[denom == 0] = 1.0
                x_all = x_all / denom

            x_list.append(x_all.astype(np.float32))
            l_list += [int(row['label'])] * x_all.shape[0]

        if not x_list:
            raise ValueError(
                "沒有任何 .pkl 與 info.csv 對得上；請檢查 features_path 與檔名。\n"
                f"- features_path = {features_path}\n"
                f"- info.csv 檔案數 = {len(df)}\n"
                f"- 對不到的 key（前 10 個）：{miss[:10]}"
            )

        x_array = np.concatenate(x_list, axis=0)
        labels = np.array(l_list, dtype=np.int64)

        self.dataset_array = torch.from_numpy(x_array)
        self.dataset_labels = torch.from_numpy(labels)
        print(f"[ok] loaded features -> X {self.dataset_array.shape}, y {self.dataset_labels.shape}")
        if miss:
            print(f"[warn] 有 {len(miss)} 個檔在 features_path 找不到對應的 .pkl（已略過）。")

    ########################################################

    def class_weight_df(self, df):
        """
        计算不同类别examples的不均衡度, 不同的方法对acc会有一定影响.
        :param df:
        :return: class_weight dict
        """
        label_1 = df['label'].sum()
        label_0 = len(df) - label_1
        # labels = [(df['label'] == n).sum() for n in range(2)]
        print(label_0, label_1)

        # class_weight = {0: 2*label_1 / len(df), 2*1: label_0 / len(df)}
        class_weight = {0: label_1 / len(df), 1: label_0 / len(df)}
        print(class_weight)

        return class_weight

    def df_fif_to_df_epochs(self, df_fif):
        """
        逐行读入df, 根据df的file, 在_examples.csv中选取对应的examples, 组成数据集.
        :param df_fif:
        :return:
        """
        df_examples_all = pd.read_csv(self.epochs_file, index_col=0, dtype={'sn_str': str})

        df_list = []
        for index in df_fif.index:
            # examples of fif
            df_examples_of_fif = df_examples_all[df_examples_all['fif_idx'] == index].copy()

            # # CL合格的
            # df_cl = df_examples_of_fif[df_examples_of_fif['label'] == df_examples_of_fif['y_star']]
            df_list.append(df_examples_of_fif)

        df = pd.concat(df_list, axis=0)
        return df

    ########################################################

    def mark_fif_info(self, df_fif, log_index=-1):
        """
        使用多个model的预测结果, 标注相关example字段.
        :param df_fif:
        :param log_index:
        :return:
        """
        df_info = pd.read_csv(self.info_file, index_col=0)

        df_log = pd.read_csv(self.log.log_file)
        last_model_index = df_log.index[log_index]

        for index in df_fif.index:
            col = 'M' + str(self.group)
            df_info.loc[index, col] = int(last_model_index)
            # df_info.loc[index, 'step'] = self.step + 1

        df_info.to_csv(self.info_file)
        df_info.to_csv(self.info_bak_file)

    ########################################################

    def rate_fif(self, df_fif_to_rate, epochs_file_name, dataset='', log_index=0):
        """
        计算rate, 并据此计算 Acc 与 Precision
        :param df_fif_to_rate: 待计算rate的DF或DF切片
        :param epochs_file_name:
        :param dataset: 'testing' | 'others'
        :param log_index:
        :return:
        """
        df_info = pd.read_csv(self.info_file, index_col=0)
        df_epochs = pd.read_csv(epochs_file_name, index_col=0)

        rate_list = []
        label_list = []

        for index in df_fif_to_rate.index:
            row = df_fif_to_rate.loc[index]
            file_path = row['file'].strip()
            file = os.path.split(file_path)[-1]

            label = row['label']
            label_list.append(label)

            # 此 fif 文件对应的 epochs
            df_epochs_of_fif = df_epochs[df_epochs['file'].str.contains(file, regex=False)].copy()

            rate = df_epochs_of_fif['Pred'].mean()
            rate_list.append(rate)

            df_epochs.loc[df_epochs_of_fif.index, 'Rate'] = rate  # 记录 only
            df_epochs.loc[df_epochs_of_fif.index, 'Dataset'] = dataset

            df_fif_to_rate.loc[index, 'Rate'] = rate  # for df_plot_(df)

            df_info.loc[index, 'Rate'] = rate

        df_epochs.to_csv(epochs_file_name)
        df_info.to_csv(self.info_file)

        # 计算Acc
        y_pred = np.array(rate_list) > 0.5  # 区分 P&N
        y_true = np.array(label_list)

        from sklearn.metrics import classification_report

        rpt = classification_report(y_true, y_pred, output_dict=True)
        acc = rpt['accuracy']
        TPR = rpt['1']['precision']
        recall = rpt['1']['recall']

        # logging #
        logging.info('%s Acc: \t%.4f' % (dataset, acc))
        logging.info('%s TPR: \t%.4f' % (dataset, TPR))
        logging.info('%s recall: \t%.4f' % (dataset, recall))
        if dataset == 'Testing':
            logging.info('Testing Index: \t%s' % (str(list(df_fif_to_rate.index.to_numpy())),))

        def find_blank_(df):
            """
            寻咋空白区域的笨办法
            :param df:
            :return:
            """
            h = 10 + 1
            w = df.index.to_numpy()[-1] + 1
            blank = np.zeros((h, w))
            x = df.index.to_numpy()
            y = (df[rate_col].to_numpy() * 10).astype('int16')
            blank[y, x] = 1

            # 快速搜索
            default = [(df.index.to_numpy()[0] + 10, 9),
                       (df.index.to_numpy()[0] + 10, 4),
                       (df.index.to_numpy()[-1] - 15, 9),
                       (df.index.to_numpy()[-1] - 15, 4)
                       ]
            for (x0, y0) in default:
                total_sum = blank[y0 - 3:y0, x0:x0 + 15].sum()
                if total_sum == 0:
                    return x0, y0 / 10 - 0.1

            # 遍历
            for x0 in range(df.index.to_numpy()[0] + 10, df.index.to_numpy()[-1] - 15, 5):
                for y0 in range(9, 3, -1):
                    print(y0, x0)
                    total = blank[y0 - 3:y0, x0:x0 + 15].sum()
                    if total == 0:
                        return x0, y0 / 10 - 0.1
            return 20, 0.8

        def df_plot_(df):
            x, y = find_blank_(df)

            df0 = df[df['label'] == 0]
            df1 = df[df['label'] == 1]
            df_padding = df[df['file'].str.contains('padding', regex=False)]

            plt.figure(figsize=(10, 6))

            plt.plot(df0[rate_col], 'o', label=class_names[0])
            plt.plot(df1[rate_col], 'o', label=class_names[1])
            plt.plot(df_padding[rate_col] + 0.02, '|g', label='padding')

            plt.legend()
            plt.ylim([-0.1, 1.1])
            plt.title(dataset)
            plt.grid(axis='y', linestyle='--')
            plt.yticks([0.0, 0.1, 0.4, 0.5, 0.6, 0.9, 1.0])

            # 显示Acc & TPR
            # plt.text(x, y, 'Acc=%.4f \nTPR=%.4f \nrecall=%.4f \nTotal=%d' % (acc, TPR, recall, len(df)))
            plt.text(x, y, 'Acc=%.4f' % (acc,))
            plt.text(x, y - 0.06, 'TPR=%.4f' % (TPR,))
            plt.text(x, y - 0.06 * 2, 'recall=%.4f' % (recall,))
            plt.text(x, y - 0.06 * 3, 'Total=%d' % (len(df),))

            if dataset != 'Black_List':
                # 显示错误的CTL
                for idx_ in df0[df0[rate_col] >= 0.5].index:
                    plt.text(idx_ + 1, df0.loc[idx_, rate_col], '%d' % (idx_,))

                # 显示错误的~CTL
                for idx_ in df1[df1[rate_col] <= 0.5].index:
                    plt.text(idx_ + 1, df1.loc[idx_, rate_col], '%d' % (idx_,))

            # Black List
            if dataset == 'Black_List':
                # 正确的CTL
                for idx_ in df0[df0[rate_col] < 0.5].index:
                    plt.text(idx_ + 1, df0.loc[idx_, rate_col], '%d' % (idx_,))

                # 正确的~CTL
                for idx_ in df1[df1[rate_col] > 0.5].index:
                    plt.text(idx_ + 1, df1.loc[idx_, rate_col], '%d' % (idx_,))

            # save
            df_log = pd.read_csv(self.log.log_file)
            fig_file_path = df_log['h5_csv'].iloc[log_index]
            fig_file = os.path.split(fig_file_path)[-1]

            plt.savefig(self.log.fig_path + fig_file + '_' + dataset + '.png')
            # plt.savefig(self.log.fig_path + fig_file + '_' + dataset + '.svg')  # 保存为矢量图形, 无极放大

        df_plot_(df_fif_to_rate)

        def df_err_(df):
            df0 = df[df['label'] == 0]
            df1 = df[df['label'] == 1]

            fn = df0[df0[rate_col] > 0.5].index.to_list()
            fp = df1[df1[rate_col] < 0.5].index.to_list()

            tn = df0[df0[rate_col] < 0.5].index.to_list()
            tp = df1[df1[rate_col] > 0.5].index.to_list()

            return str(fn + fp), str(tn + tp)

        return acc, TPR, recall, df_err_(df_fif_to_rate)

    ########################################################

    def step_set(self, step=0):
        df_info = pd.read_csv(self.info_file, index_col=0)
        df_epochs_sn = pd.read_csv(self.epochs_file, index_col=0, dtype={'sn_str': str})
        # df_epochs_offset = pd.read_csv(self.epochs_offset_group_file, index_col=0, dtype={'sn_str': str})
        # df_examples_offset = pd.read_csv(self.examples_offset_group_file, index_col=0, dtype={'sn_str': str})

        df_info['step'] = step
        df_epochs_sn['step'] = step
        # df_epochs_offset['step'] = step
        # df_examples_offset['step'] = step

        # df_info.to_csv(self.info_file)
        df_epochs_sn.to_csv(self.epochs_file)
        # df_epochs_offset.to_csv(self.epochs_offset_group_file)
        # df_examples_offset.to_csv(self.examples_offset_group_file)

        self.step = step

    def evaluate_all_epochs_of_dataset_CL(self):
        """

        :return:
        """
        from numpy import nan

        df_examples = pd.read_csv(self.epochs_file, index_col=0)

        # 计算各个类别self-confidence的均值, 作为阈值
        t = [0, 0]
        t[0] = df_examples[df_examples['label'] == 0]['Prob0'].mean()
        t[1] = df_examples[df_examples['label'] == 1]['Prob1'].mean()

        # 候选预测输出
        df_examples['y_0'] = df_examples['Prob0'] > t[0]
        df_examples['y_1'] = df_examples['Prob1'] > t[1]

        df_examples['y_star'] = nan

        df_y_0 = df_examples[df_examples['y_0']]
        df_examples.loc[df_y_0.index, 'y_star'] = 0

        df_y_1 = df_examples[df_examples['y_1']]
        df_examples.loc[df_y_1.index, 'y_star'] = 1  # 可能collision

        # 消除存在的collision
        df_y_collision = df_examples[df_examples['y_0'] & df_examples['y_1']]
        y_collision_pred = df_y_collision[['Prob0', 'Prob1']].to_numpy().argmax(axis=1)
        df_examples.loc[df_y_collision.index, 'y_star'] = y_collision_pred

        df_examples.to_csv(self.epochs_file)

    def evaluate_all_fifs_of_dataset_CL(self):
        """

        :return:
        """
        df_dataset_info = pd.read_csv(self.info_file, index_col=0)
        df_examples = pd.read_csv(self.epochs_file, index_col=0)

        # 计算 rate
        for idx in df_dataset_info.index:
            df_epochs_sel = df_examples[df_examples['fif_idx'] == idx]

            rate = df_epochs_sel['y_star'].to_numpy().mean()
            df_dataset_info.loc[idx, 'Prob1'] = rate
            df_dataset_info.loc[idx, 'Prob0'] = 1 - rate

        fif_t = [0, 0]
        for label in [0, 1]:
            fif_t[label] = df_dataset_info['Prob%d' % (label,)].mean()

        df_dataset_info['y_0'] = df_dataset_info['Prob0'] > fif_t[0]
        df_dataset_info['y_1'] = df_dataset_info['Prob1'] > fif_t[1]

        df_y_0 = df_dataset_info[df_dataset_info['y_0']]
        df_dataset_info.loc[df_y_0.index, 'y_star'] = 0

        df_y_1 = df_dataset_info[df_dataset_info['y_1']]
        df_dataset_info.loc[df_y_1.index, 'y_star'] = 1

        df_y_collision = df_dataset_info[df_dataset_info['y_0'] & df_dataset_info['y_1']]
        y_collision_pred = df_y_collision[['Prob0', 'Prob1']].to_numpy().argmax(axis=1)
        df_dataset_info.loc[df_y_collision.index, 'y_star'] = y_collision_pred

        df_dataset_info.to_csv(self.info_file)

    ########################################################

    def evaluate_lpso_with_k_fold_feature(self, n_splits_fold=5, random_state=30,
                                          n_splits_test_set=5, random_state_test=30):
        """

        :param n_splits_fold:
        :param random_state:
        :param n_splits_test_set:
        :param random_state_test:
        :return:
        """
        print('>>', 'evaluate_lpso_with_k_fold_feature()')

        df_fif_all = pd.read_csv(self.info_file, index_col=0)
        df_fif_bl = df_fif_all[df_fif_all['Black_List']]
        df_fif_wl = df_fif_all[~df_fif_all['Black_List']]

        # 自定义 StratifiedKFoldDF 实例化
        skf_test = StratifiedKFoldDF(n_splits=n_splits_test_set, shuffle=True,
                                     random_state=random_state_test)

        self.split_random_state = random_state_test

        # 1 分割白名单 fif 出 df_fif_subset & df_fif_test
        # for index in df_fif_all.index:
        #
        #     logging.info('%d of Group %d' % (index, self.group))
        #
        #     df_fif_test = df_fif_all[df_fif_all.index == index]
        #     df_fif_subset = df_fif_all[df_fif_all.index != index]
        #
        #     self.df_fif_test = df_fif_test

        model_indices_for_bl = []

        # 1 分割白名单 fif 出 df_fif_subset & df_fif_test
        for k, (df_fif_subset, df_fif_test) in enumerate(skf_test.split(df_fif_wl)):
            print(df_fif_test.index.to_numpy())

            # 1.1 对 df_fif_subset 执行KFold
            model_indices = self.train_with_k_fold_feature(df_fif_subset,
                                                           n_splits=n_splits_fold,
                                                           random_state=random_state + k)

            # 1.2 使用 n_splits_fold 个模型, 对测试集给出预测
            # self.predict_examples_CL(self.df_fif_to_df_examples_offset_group(df_fif_test), n_splits=n_splits_fold, model_indices=model_indices)
            self.predict_epochs_CL(self.df_fif_to_df_epochs(df_fif_test), n_splits=n_splits_fold,
                                   model_indices=model_indices)

            model_indices_for_bl += model_indices

        # 2 使用 n_splits_fold*n_splits_test_set 个模型, 对BL集给出预测
        self.predict_epochs_CL(self.df_fif_to_df_epochs(df_fif_bl), n_splits=n_splits_fold,
                               model_indices=model_indices_for_bl)

        # # 3 使用n_splits个模型, 对数据集(含测试集)给出预测
        # self.predict_epochs(self.df_fif_to_df_epochs_offset_group(df_fif_all), n_splits=n_splits_fold, model_indices=model_indices)
        # self.rate_fif(df_fif_subset, dataset='Subset')
        # # _, _, _, bl_err = self.rate_fif(df_fif_bl, dataset='Black_List')
        # acc, TPR, recall, test_err = self.rate_fif(df_fif_test, dataset='Testing')
        #
        # # self.append_evaluate_file(acc, TPR, recall, test_err[0], bl_err[1])
        # self.append_evaluate_file(acc, TPR, recall, test_err[0], None)
        # self.mark_fif_info(df_fif_test, log_index=-1)
        #
        # logging.info('\n')

    def train_with_k_fold_feature(self, df_fif_subset, n_splits=5, random_state=0, P=1):
        """

        :param df_fif_subset:
        :param n_splits: k
        :param random_state: k分割的随机数, 固定该值可以重复实验
        :param P: KFold重复次数
        :return: index of P*k models
        """
        print('>>', 'train_with_k_fold_feature()')

        model_indices = []
        self.sn_kfold = 0

        # 对 df_fif_subset 执行KFold, 重复P次
        for p in range(P):

            # 自定义 StratifiedKFoldDF 实例化
            skf = StratifiedKFoldDF(n_splits=n_splits, shuffle=True, random_state=random_state + p)
            for df_fif_train, df_fif_val in skf.split(df_fif_subset):
                # self.df_train = self.df_fif_to_df_examples_CL(df_fif_train)
                # self.df_val = self.df_fif_to_df_examples_CL(df_fif_val)

                self.df_train = self.df_fif_to_df_epochs(df_fif_train)
                self.df_val = self.df_fif_to_df_epochs(df_fif_val)

                print(df_fif_train.index.to_numpy())
                print(df_fif_val.index.to_numpy())

                # 开始当前KFold的第k个训练
                self.fit_method()

                # 记录当前训练所得模型的索引
                model_indices.append(self.log.get_latest_model_index())
                self.sn_kfold += 1

        return model_indices

    ########################################################

    def logging_dataset_info(self):
        df_dataset_info = pd.read_csv(self.info_file, index_col=0)

        # 欄名清洗（防 BOM/空白）
        df_dataset_info.columns = df_dataset_info.columns.str.replace('\ufeff', '', regex=False).str.strip()

        # 必要欄位保底檢查
        for col in ['label', 'Black_List']:
            if col not in df_dataset_info.columns:
                raise ValueError(f"logging_dataset_info(): '{col}' column missing in {self.info_file}. "
                                f"Rebuild the info file first.")

        # 黑/白名單切分（黑名單可能為空，要能優雅處理）
        df_bl = df_dataset_info[df_dataset_info['Black_List']]
        df_wl = df_dataset_info[~df_dataset_info['Black_List']]

        if len(df_bl) == 0:
            logging.info('Black List Number: 0/0')
            logging.info('Black List Index N: []')
            logging.info('Black List Index P: []')
        else:
            df_bl_n = df_bl[df_bl['label'] == 0]
            df_bl_p = df_bl[df_bl['label'] == 1]
            logging.info('Black List Number: %d/%d' % (len(df_bl_n), len(df_bl_p)))
            logging.info('Black List Index N: %s' % (json.dumps(df_bl_n.index.to_list()),))
            logging.info('Black List Index P: %s' % (json.dumps(df_bl_p.index.to_list()),))

        logging.info('White List Number: %d' % (len(df_wl),))
        logging.info('White List Index: %s' % (json.dumps(df_wl.index.to_list()),))


    def collect_dataset_info(self):
        """
        导入合格验证文件部分信息, 历并随机分割数据集文件,
        基础数据目录存盘为 *_info.csv.
        :return:
        """

        # 用于生成DataFrame(字典生成方法)
        df_dict = {'file': [], 'class_name': [], 'label': [], 'n_times': []}

        # 收集数据集, 存入DataFrame
        for class_name in class_names:
            files_list = []
            for root, _, files in os.walk(os.path.join(base_path, class_paths[class_name])):
                for file in files:
                    file_path = os.path.join(root, file)

                    raw = io.Raw(file_path, preload=False, verbose=False)
                    if len(raw.info['ch_names']) < self.eeg_channel_num:
                        continue

                    # EC | EO
                    if not file.startswith(tuple(k.upper() for k in keyword)):
                        continue

                    # file_path = file_path.ljust(75)     # 补空格, 目的在于csv文件的可读性.
                    files_list.append(file_path)
                    print(file_path)

                    df_dict['n_times'].append(raw.n_times)

            df_dict['file'] += files_list
            df_dict['class_name'] += [class_name] * len(files_list)
            df_dict['label'] += [class_labels[class_name]] * len(files_list)

        df_dataset_info = pd.DataFrame(df_dict, index=[n for n in range(len(df_dict['file']))])

        # df_dataset_info['Black_List'] = False
        # for index in df_dataset_info.index:
        #     if index in black_list:
        #         df_dataset_info.loc[index, 'Black_List'] = True

        df_dataset_info.to_csv(self.info_file)

        # self.logging_dataset_info()

    def collect_dataset_info_with_black_list(self):
        """
        导入合格验证文件部分信息, 历并随机分割数据集文件,
        基础数据目录存盘为 *_info.csv.
        :return:
        """

        # 用于生成DataFrame(字典生成方法)
        df_dict = {'file': [], 'class_name': [], 'label': [], 'n_times': []}

        # 收集数据集, 存入DataFrame
        for class_name in class_names:
            files_list = []
            for root, _, files in os.walk(os.path.join(base_path, class_paths[class_name])):
                for file in files:
                    file_path = os.path.join(root, file)

                    # EC | EO 篩檔名
                    if not file.startswith(tuple(k.upper() for k in keyword)):
                        continue

                    try:
                        if file.endswith('-epo.fif') or file.endswith('-epo.fif.gz'):
                            # epochs 檔
                            epochs = mne.read_epochs(file_path, preload=False, verbose=False)
                            if len(epochs.info['ch_names']) < self.eeg_channel_num:
                                continue
                            n_unit = len(epochs)              # 用「epoch 數」作為後續單位
                        else:
                            # raw 檔
                            raw = mne.io.read_raw_fif(file_path, preload=False, verbose=False)
                            if len(raw.info['ch_names']) < self.eeg_channel_num:
                                continue
                            n_unit = raw.n_times              # raw 用時間點數

                    except Exception as e:
                        print(f"[skip] {file_path} -> {e}")
                        continue

                    files_list.append(file_path)
                    print(file_path)
                    df_dict['n_times'].append(int(n_unit))     # 對 epochs 其實是 n_epochs

            df_dict['file'] += files_list
            df_dict['class_name'] += [class_name] * len(files_list)
            df_dict['label'] += [class_labels[class_name]] * len(files_list)


        # 組好 DataFrame 後，先做健檢
        df_dataset_info = pd.DataFrame(df_dict)

        # 強制標準欄位存在
        required = ['file', 'class_name', 'label', 'n_times']
        missing = [c for c in required if c not in df_dataset_info.columns]
        if missing:
            raise ValueError(f"collect_dataset_info_with_black_list(): missing columns {missing}. "
                            f"Check class_labels/class_names and the file walking logic.")

        # 類別轉型，避免後續比較時出問題
        df_dataset_info['label'] = df_dataset_info['label'].astype(int)

        # 先全部標 False；之後你的 vote_and_save() 會再改 True
        df_dataset_info['Black_List'] = False

        # 寫檔
        df_dataset_info.to_csv(self.info_file, index=True)

        # 立刻讀回來一次並清洗欄名（去掉 BOM 和首尾空白），再寫回，避免隱形欄名
        tmp = pd.read_csv(self.info_file, index_col=0)
        tmp.columns = tmp.columns.str.replace('\ufeff', '', regex=False).str.strip()
        tmp.to_csv(self.info_file, index=True)

        print('info_file written:', self.info_file)
        print('info_file columns:', list(tmp.columns))

        self.logging_dataset_info()

    def collect_dataset_info_with_qualification(self):
        """
        导入合格验证文件部分信息, 历并随机分割数据集文件,
        基础数据目录存盘为 *_info.csv.
        :return:
        """

        # 合格验证文件
        df_info_q = pd.read_csv(qualify_info_file, index_col=0)

        # 用于生成DataFrame(字典生成方法)
        df_dict = {'file': [], 'class_name': [], 'label': [], 'n_times': []}

        # 收集数据集, 存入DataFrame
        for class_name in class_names:
            files_list = []
            for root, _, files in os.walk(os.path.join(base_path, class_paths[class_name])):
                for file in files:
                    file_path = os.path.join(root, file)

                    raw = io.Raw(file_path, preload=False, verbose=False)
                    if len(raw.info['ch_names']) < self.eeg_channel_num:
                        continue

                    # EC | EO
                    if not file.startswith(tuple(k.upper() for k in keyword)):
                        continue

                    file_path = file_path.ljust(75)  # 补空格, 目的在于csv文件的可读性.
                    files_list.append(file_path)
                    print(file_path)

                    df_dict['n_times'].append(raw.n_times)

            df_dict['file'] += files_list
            df_dict['class_name'] += [class_name] * len(files_list)
            df_dict['label'] += [class_labels[class_name]] * len(files_list)

        df_dataset_info = pd.DataFrame(df_dict, index=[n for n in range(len(df_dict['file']))])

        # 导入合格验证文件部分信息
        df_dataset_info['Acc'] = df_info_q['Acc']
        df_dataset_info['Qualify'] = df_info_q['Pred']

        # 修正
        for index in black_list:
            df_dataset_info.loc[index, 'Qualify'] = 1 - df_dataset_info.loc[index, 'label']

        # # 修正
        # for index in white_list:
        #     df_dataset_info.loc[index, 'Qualify'] = df_dataset_info.loc[index, 'label']

        df_qualified = df_dataset_info[df_dataset_info['label'] == df_dataset_info['Qualify']]
        print('df_qualified:', df_qualified.shape)

        df_dataset_info['dataset'] = 'unqualified'
        df_dataset_info.loc[df_qualified.index, 'dataset'] = 'qualified'

        # 'Black_List' 与 'unqualified' 等效
        df_dataset_info['Black_List'] = True
        df_dataset_info.loc[df_qualified.index, 'Black_List'] = False

        df_dataset_info.to_csv(self.info_file)

        self.logging_dataset_info()

    ########################################################

    def append_evaluate_file(self, acc, tpr, recall, test_err, bl_err):
        """
        在log文件尾部追加记录(添加一行). 比添加一列要麻烦!
        如果log文件不存在, 则新建一个.
        :return:
        """
        from datetime import datetime

        d = {'date_time': datetime.now().strftime('%Y-%m-%d/%H:%M:%S'),

             'Acc': acc,
             'TPR': tpr,
             'Recall': recall,
             'Test_FN/FP': test_err,
             'BL_FN/FP': bl_err}

        df_evaluate_new = pd.DataFrame(d, index=[0])

        if os.path.exists(self.evaluate_file):
            df_evaluate = pd.read_csv(self.evaluate_file, index_col=0)
            df_evaluate = pd.concat([df_evaluate, df_evaluate_new], axis=0, ignore_index=True, sort=False)  # 必须用返回值赋值
        else:
            df_evaluate = df_evaluate_new

        df_evaluate.to_csv(self.evaluate_file)

    def summary(self):
        import json

        df = pd.read_csv(self.evaluate_file, index_col=0)

        acc = df['Acc'].mean()
        tpr = df['TPR'].mean()
        recall = df['Recall'].mean()
        print(acc, tpr, recall)

        df1 = df['Test_FN/FP'].to_list()

        test_err = []
        for item in df1:
            test_err += json.loads(item)

        test_err_cnt = [(item, test_err.count(item)) for item in set(test_err)]
        test_err_cnt.sort(key=lambda x: x[1], reverse=True)
        print('test_err_cnt:\n', test_err_cnt)

        df2 = df['BL_FN/FP'].to_list()
        bl_err = []
        for item in df2:
            bl_err += json.loads(item)

        bl_err_cnt = [(item, bl_err.count(item)) for item in set(bl_err)]
        bl_err_cnt.sort(key=lambda x: x[1], reverse=True)
        print('bl_err_cnt:\n', bl_err_cnt)

    def save_info_as(self, sn=0):
        df = pd.read_csv(self.info_file, index_col=0)
        info_new_file = self.info_file[:-4] + '_%d.csv' % (sn,)
        df.to_csv(info_new_file)

    def save_examples_info_as(self, sn=0):
        df = pd.read_csv(self.examples_offset_group_file, index_col=0)
        info_new_file = self.examples_offset_group_file[:-4] + '_%d.csv' % (sn,)
        df.to_csv(info_new_file)

    def save_epochs_info_as(self, sn=0):
        df = pd.read_csv(self.epochs_file, index_col=0)
        info_new_file = self.epochs_file[:-4] + '_%d.csv' % (sn,)
        df.to_csv(info_new_file)

    def select_black_list(self):
        df_vote = None
        n_files = 7
        pred_cols = ['P%d' % (sn,) for sn in range(n_files)]
        for sn in range(n_files):
            df_file_n = self.epochs_offset_group_file[:-4] + '_%d.csv' % (sn,)
            df_n = pd.read_csv(df_file_n, index_col=0)

            if df_vote is None:
                df_vote = df_n[['upper_limit', 'fif_idx', 'ID', 'label']].copy()

            df_vote[pred_cols[sn]] = df_n['y_star']

        for index in df_vote.index:
            row = df_vote.loc[index]
            pred_list = row[pred_cols].to_list()
            df_vote.loc[index, 'Vote'] = max(set(pred_list), key=pred_list.count)

        df_vote.to_csv(self.epochs_offset_group_file[:-4] + '_vote.csv')

    ########################################################
    # Co-teaching train
    ########################################################

    def define_adam_learning_rate(self):
        n_epoch = eeg_config['epochs']
        learning_rate = eeg_config['lr_init']
        epoch_decay_start = eeg_config['epoch_decay_start']

        # Adjust learning rate and betas for Adam Optimizer
        mom1 = 0.9
        mom2 = 0.1
        self.alpha_plan = [learning_rate] * n_epoch
        self.beta1_plan = [mom1] * n_epoch
        for i in range(epoch_decay_start, n_epoch):
            self.alpha_plan[i] = float(n_epoch - i) / (n_epoch - epoch_decay_start) * learning_rate
            self.beta1_plan[i] = mom2

    def adjust_adam_learning_rate(self, optimizer, epoch):
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.alpha_plan[epoch]
            param_group['betas'] = (self.beta1_plan[epoch], 0.999)  # Only change beta1

    def define_drop_rate_schedule(self, n_epoch):
        # define drop rate schedule
        forget_rate = eeg_config['forget_rate']
        num_gradual = eeg_config['num_gradual']  # Tk
        exponent = eeg_config['exponent']  # c in Tc

        self.rate_schedule = np.ones(n_epoch) * forget_rate
        self.rate_schedule[:num_gradual] = np.linspace(0, forget_rate ** exponent, num_gradual)

        return self.rate_schedule

    def gen_forgetrate_schedule(self, n_epoch):
        # define drop rate schedule
        forget_rate = eeg_config['forget_rate']
        num_gradual = eeg_config['num_gradual']  # Tk
        exponent = eeg_config['exponent']  # c in Tc

        self.rate_schedule = np.ones(n_epoch) * forget_rate
        self.rate_schedule[:num_gradual] = np.linspace(0, forget_rate, num_gradual)
        self.rate_schedule[num_gradual:] = np.linspace(forget_rate, 2 * forget_rate, n_epoch - num_gradual)

        return self.rate_schedule

    def adjust_learning_rate(self, optimizer, init_lr, epoch, epochs):
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

    def build_train_loader(self, batch_size=64):
        """
        :return:
        """

        df_epochs = pd.read_csv(self.epochs_file)
        # self.load_feature_data_to_mem(normalize=eeg_config['data_normalize'])

        dataset = eeg_config['dataset_train']

        train_dataset = dataset(self.df_train,  # df_epochs,#
                                labels=self.dataset_labels,
                                dataset_array=self.dataset_array,
                                # device=device,
                                # channels=self.channel_factor
                                )

        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=0,
                                                   pin_memory=True,
                                                   sampler=None,
                                                   drop_last=True)

        return train_loader

    def train_co_teaching_plus(self):
        """
        Co_teaching
        :return:
        """
        import time

        epochs = eeg_config['epochs']
        batch_size = eeg_config['batch_size']
        init_epoch = eeg_config['init_epoch']

        # select model
        net = eeg_config['net']
        model1 = net(self.n_classes, eeg_config['feature_dim'], eeg_config['fc_dim'])
        model2 = net(self.n_classes, eeg_config['feature_dim'], eeg_config['fc_dim'])
        print(model1)

        model1.cuda(device)
        model2.cuda(device)

        # define optimizer
        optimizer1 = torch.optim.SGD(model1.parameters(),
                                     eeg_config['lr_init'],
                                     momentum=0.9,
                                     weight_decay=eeg_config['weight_decay'])

        optimizer2 = torch.optim.SGD(model2.parameters(),
                                     eeg_config['lr_init'],
                                     momentum=0.9,
                                     weight_decay=eeg_config['weight_decay'])

        # define drop rate schedule
        rate_schedule = self.gen_forgetrate_schedule(epochs)

        # 自动生成模型文件名/训练历史文件名
        model_file = self.log.open(model1.__class__.__name__)
        # class_weight_training = self.class_weight_df(self.df_train)

        train_loader = self.build_train_loader(batch_size=batch_size)

        for epoch in range(epochs):
            print("\nEpoch {0}/{1}".format(str(epoch + 1), str(epochs)))

            # 0. set learning rate every epoch
            lr1 = adjust_learning_rate(optimizer1, eeg_config['lr_init'], epoch, epochs)
            lr2 = adjust_learning_rate(optimizer2, eeg_config['lr_init'], epoch, epochs)
            # print(lr1, lr2)

            t0 = time.time()

            losses1_meter = AverageMeter('Loss1', ':.6f')
            losses2_meter = AverageMeter('Loss2', ':.6f')
            acc1_meter = AccuracyMeter('Acc1', ':.4f')
            acc2_meter = AccuracyMeter('Acc2', ':.4f')

            lossesA_meter = AverageMeter('LossA', ':.6f')
            lossesB_meter = AverageMeter('LossB', ':.6f')

            remember_meter = AverageMeter('Remember', ':.2f')

            model1.train()
            model2.train()

            # 1. 逐个batch进行训练
            for batch_idx, (xs, labels) in enumerate(train_loader):
                xs = xs.cuda(device, non_blocking=True)
                labels = labels.cuda(device, non_blocking=True)

                # Forward + Backward + Optimize
                logits1 = model1(xs)
                logits2 = model2(xs)

                loss_1_test = F.cross_entropy(logits1, labels)
                loss_2_test = F.cross_entropy(logits2, labels)

                if epoch < init_epoch:  # Co-teaching, warm up
                    loss_1, loss_2, num_remember = loss_coteaching_m(logits1,
                                                                     logits2,
                                                                     labels,
                                                                     forget_rate=rate_schedule[epoch])
                else:   # Co-teaching++
                    loss_1, loss_2, num_remember = loss_coteaching_plus_m5(logits1,
                                                                           logits2,
                                                                           labels,
                                                                           forget_rate=rate_schedule[epoch],
                                                                           step=epoch * batch_idx
                                                                           )

                optimizer1.zero_grad()
                loss_1.backward()
                optimizer1.step()

                optimizer2.zero_grad()
                loss_2.backward()
                optimizer2.step()

                # 记录
                if num_remember == xs.size(0):
                    losses1_meter.update(loss_1_test.item(), xs.size(0))
                    losses2_meter.update(loss_2_test.item(), xs.size(0))
                else:
                    loss_1_test = (loss_1_test.item() * xs.size(0) - loss_1.item() * num_remember) / (xs.size(0) - num_remember)
                    loss_2_test = (loss_2_test.item() * xs.size(0) - loss_2.item() * num_remember) / (xs.size(0) - num_remember)
                    losses1_meter.update(loss_1_test, xs.size(0) - num_remember)
                    losses2_meter.update(loss_2_test, xs.size(0) - num_remember)

                acc1_meter.update(logits1.detach().cpu().numpy(), labels.cpu().numpy())
                acc2_meter.update(logits2.detach().cpu().numpy(), labels.cpu().numpy())

                lossesA_meter.update(loss_1.item(), num_remember)
                lossesB_meter.update(loss_2.item(), num_remember)

                remember_meter.update(num_remember, 1)

            print(remember_meter)
            print(losses1_meter, losses2_meter, lossesA_meter, lossesB_meter)
            print(acc1_meter, acc2_meter)
            print(time.time() - t0)

            # 2. 记录
            self.log.monitor.update(acc1=acc1_meter.get_value(),
                                    acc2=acc2_meter.get_value(),
                                    lossA=lossesA_meter.get_value(),
                                    lossB=lossesB_meter.get_value(),
                                    loss1=losses1_meter.get_value(),
                                    loss2=losses2_meter.get_value()
                                    )

            # 3. save thr last ten models
            if epoch >= 190:
                torch.save(model1.state_dict(), model_file[:-3] + '_A_%d.pt' % (epoch, ))
                torch.save(model2.state_dict(), model_file[:-3] + '_B_%d.pt' % (epoch, ))

        # -1. save
        self.log.close()

    def train_co_teaching_plus_adam(self):
        """
        Co_teaching
        :return:
        """
        import time

        epochs = eeg_config['epochs']
        batch_size = eeg_config['batch_size']
        init_epoch = eeg_config['init_epoch']

        # select model
        net = eeg_config['net']
        model1 = net(self.n_classes, eeg_config['feature_dim'], eeg_config['fc_dim'])
        model2 = net(self.n_classes, eeg_config['feature_dim'], eeg_config['fc_dim'])
        print(model1)

        model1.cuda(device)
        model2.cuda(device)

        # define optimizer
        optimizer1 = torch.optim.Adam(model1.parameters(),
                                      lr=eeg_config['lr_init'])

        optimizer2 = torch.optim.Adam(model2.parameters(),
                                      lr=eeg_config['lr_init'])

        self.define_adam_learning_rate()

        # define drop rate schedule
        rate_schedule = self.gen_forgetrate_schedule(epochs)

        # 自动生成模型文件名/训练历史文件名
        model_file = self.log.open(model1.__class__.__name__)
        # class_weight_training = self.class_weight_df(self.df_train)

        train_loader = self.build_train_loader(batch_size=batch_size)

        for epoch in range(epochs):
            print("\nEpoch {0}/{1}".format(str(epoch + 1), str(epochs)))

            # 0. set learning rate every epoch
            self.adjust_adam_learning_rate(optimizer1, epoch)
            self.adjust_adam_learning_rate(optimizer2, epoch)

            t0 = time.time()

            losses1_meter = AverageMeter('Loss1', ':.6f')
            losses2_meter = AverageMeter('Loss2', ':.6f')
            acc1_meter = AccuracyMeter('Acc1', ':.4f')
            acc2_meter = AccuracyMeter('Acc2', ':.4f')

            lossesA_meter = AverageMeter('LossA', ':.6f')
            lossesB_meter = AverageMeter('LossB', ':.6f')

            remember_meter = AverageMeter('Remember', ':.2f')

            model1.train()
            model2.train()

            # 1. 逐个batch进行训练
            for batch_idx, (xs, labels) in enumerate(train_loader):
                xs = xs.cuda(device, non_blocking=True)
                labels = labels.cuda(device, non_blocking=True)

                # Forward + Backward + Optimize
                logits1 = model1(xs)
                logits2 = model2(xs)

                loss_1_test = F.cross_entropy(logits1, labels)
                loss_2_test = F.cross_entropy(logits2, labels)

                if epoch < init_epoch:  # Co-teaching, warm up
                    loss_1, loss_2, num_remember = loss_coteaching_m(logits1,
                                                                     logits2,
                                                                     labels,
                                                                     forget_rate=rate_schedule[epoch])
                else:   # Co-teaching++
                    loss_1, loss_2, num_remember = loss_coteaching_plus_m5(logits1,
                                                                           logits2,
                                                                           labels,
                                                                           forget_rate=rate_schedule[epoch],
                                                                           step=epoch * batch_idx
                                                                           )

                optimizer1.zero_grad()
                loss_1.backward()
                optimizer1.step()

                optimizer2.zero_grad()
                loss_2.backward()
                optimizer2.step()

                # 记录
                if num_remember == xs.size(0):
                    losses1_meter.update(loss_1_test.item(), xs.size(0))
                    losses2_meter.update(loss_2_test.item(), xs.size(0))
                else:
                    loss_1_test = (loss_1_test.item() * xs.size(0) - loss_1.item() * num_remember) / (xs.size(0) - num_remember)
                    loss_2_test = (loss_2_test.item() * xs.size(0) - loss_2.item() * num_remember) / (xs.size(0) - num_remember)
                    losses1_meter.update(loss_1_test, xs.size(0) - num_remember)
                    losses2_meter.update(loss_2_test, xs.size(0) - num_remember)

                acc1_meter.update(logits1.detach().cpu().numpy(), labels.cpu().numpy())
                acc2_meter.update(logits2.detach().cpu().numpy(), labels.cpu().numpy())

                lossesA_meter.update(loss_1.item(), num_remember)
                lossesB_meter.update(loss_2.item(), num_remember)

                remember_meter.update(num_remember, 1)

            print(remember_meter)
            print(losses1_meter, losses2_meter, lossesA_meter, lossesB_meter)
            print(acc1_meter, acc2_meter)
            print(time.time() - t0)

            # 2. 记录
            self.log.monitor.update(acc1=acc1_meter.get_value(),
                                    acc2=acc2_meter.get_value(),
                                    lossA=lossesA_meter.get_value(),
                                    lossB=lossesB_meter.get_value(),
                                    loss1=losses1_meter.get_value(),
                                    loss2=losses2_meter.get_value()
                                    )

            # 3. save thr last ten models
            if epoch >= 190:
                torch.save(model1.state_dict(), model_file[:-3] + '_A_%d.pt' % (epoch, ))
                torch.save(model2.state_dict(), model_file[:-3] + '_B_%d.pt' % (epoch, ))

        # -1. save
        self.log.close()

    def evaluate(self, net1, net2, data_loader, cost=None):
        # losses1 = AverageMeter('Loss1', ':.6f')
        acc1 = AccuracyMeter('val-Acc1', ':.6f')

        # losses2 = AverageMeter('Loss2', ':.6f')
        acc2 = AccuracyMeter('val-Acc2', ':.6f')

        net1.eval()
        net2.eval()
        with torch.no_grad():
            for x, y in data_loader:
                x, y = x.cuda(device, non_blocking=True), y.cuda(device, non_blocking=True)

                y_hat1 = net1(x)
                y_hat2 = net2(x)

                # loss = cost(y_hat, y.long())
                # losses.update(loss.item(), x[0].size(0))

                acc1.update(y_hat1.cpu().numpy(), y.cpu().numpy())
                acc2.update(y_hat2.cpu().numpy(), y.cpu().numpy())

        print(acc1, acc2)

        # return losses.get_value(), acc.get_value()
        return acc1.get_value(), acc2.get_value()

    ########################################################
    # Co-teaching predict
    ########################################################

    def evaluate_self_feature(self):
        """

        """
        print('>>', 'evaluate_self_feature()')

        df_fif_all = pd.read_csv(self.info_file, index_col=0)
        df_fif_wl = df_fif_all[~df_fif_all['Black_List']]

        self.df_train = self.df_fif_to_df_epochs(df_fif_wl)
        self.df_val = self.df_fif_to_df_epochs(df_fif_wl)

        self.fit_method()
        model_index = self.log.get_latest_model_index()

        self.predict_epochs_Co(self.df_val, model_index=model_index)

    def predict_epochs_Co(self, df_epochs_to_predict, model_index):
        """
        评估模型在指定数据集上的性能.
        :param df_epochs_to_predict:
        :param model_index: int
        :return:
        """

        df_log = pd.read_csv(self.log.log_file)

        y_probs1 = np.zeros((len(df_epochs_to_predict), 2), dtype='float32')
        y_probs2 = np.zeros((len(df_epochs_to_predict), 2), dtype='float32')

        main_name = df_log['pt_csv'].iloc[model_index]

        # over last ten epochs
        for sn in range(190, 200, 1):
            state_dict_file1 = main_name + '_A_%d.pt' % (sn, )
            state_dict_file2 = main_name + '_B_%d.pt' % (sn, )

            net = eeg_config['net']
            model1 = net(self.n_classes, eeg_config['feature_dim'], eeg_config['fc_dim'])
            model2 = net(self.n_classes, eeg_config['feature_dim'], eeg_config['fc_dim'])

            state_dict1 = torch.load(state_dict_file1)
            model1.load_state_dict(state_dict1, strict=False)
            model1.cuda(device)

            state_dict2 = torch.load(state_dict_file2)
            model2.load_state_dict(state_dict2, strict=False)
            model2.cuda(device)

            acc1 = AccuracyMeter('Acc1', ':.4f')
            acc2 = AccuracyMeter('Acc2', ':.4f')

            model1.eval()
            model2.eval()
            with torch.no_grad():
                for x, y in self.build_loader(df_epochs_to_predict):
                    x, y = x.cuda(device, non_blocking=True), y.cuda(device, non_blocking=True)

                    y_hat1 = model1(x)
                    y_hat2 = model2(x)

                    acc1.update(y_hat1.cpu().numpy(), y.cpu().numpy())
                    acc2.update(y_hat2.cpu().numpy(), y.cpu().numpy())

            print(acc1)
            print(acc2)

            logging.info('Acc1=%.4f, Acc2=%.4f' % (acc1.get_value(), acc2.get_value()))

            y_prob1 = acc1.get_proba()
            y_probs1 += y_prob1

            y_prob2 = acc2.get_proba()
            y_probs2 += y_prob2

        # y_prob1 = y_probs1 / 10
        # y_prob2 = y_probs2 / 10

        y_prob1 = y_probs1 / y_probs1.sum(axis=1, keepdims=True)  # n_splits
        y_prob2 = y_probs2 / y_probs2.sum(axis=1, keepdims=True)  # n_splits

        # self.label_epochs_CL(df_epochs_to_predict, y_prob)

        df_epochs = pd.read_csv(self.epochs_file, index_col=0)

        y_pred1 = np.argmax(y_prob1, axis=-1)

        label = df_epochs.loc[df_epochs_to_predict.index, 'label']

        df_epochs.loc[df_epochs_to_predict.index, 'Pred1'] = y_pred1
        df_epochs.loc[df_epochs_to_predict.index, 'Prob10'] = y_prob1[:, 0]
        df_epochs.loc[df_epochs_to_predict.index, 'Prob11'] = y_prob1[:, 1]
        df_epochs.loc[df_epochs_to_predict.index, 'SC1'] = y_prob1[range(len(df_epochs_to_predict)), label]

        y_pred2 = np.argmax(y_prob2, axis=-1)

        label = df_epochs.loc[df_epochs_to_predict.index, 'label']

        df_epochs.loc[df_epochs_to_predict.index, 'Pred2'] = y_pred2
        df_epochs.loc[df_epochs_to_predict.index, 'Prob20'] = y_prob2[:, 0]
        df_epochs.loc[df_epochs_to_predict.index, 'Prob21'] = y_prob2[:, 1]
        df_epochs.loc[df_epochs_to_predict.index, 'SC2'] = y_prob2[range(len(df_epochs_to_predict)), label]

        df_epochs.to_csv(self.epochs_file)

    def build_loader(self, df_epochs, batch_size=1024):
        """
        :return:
        """

        dataset = eeg_config['dataset_predict']

        test_dataset = dataset(df_epochs,
                               labels=self.dataset_labels,
                               dataset_array=self.dataset_array)

        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=0,
                                                  pin_memory=True,
                                                  sampler=None,
                                                  drop_last=False)

        return test_loader

    ########################################################
    # Co-teaching predict end
    ########################################################

    def vote_and_save(self):
        from Plot_CL_epoch_Vote_502WL_TH075 import score_plot

        epochs_file_list = []
        for n in range(0, 5, 1):
            epochs_file = self.epochs_file[:-4] + '_%s.csv' % (n, )
            epochs_file_list.append(epochs_file)

        bl0, bl1 = score_plot(epochs_file_list, th=0.75)

        df_dataset_info = pd.read_csv(self.info_file, index_col=0)
        for index in df_dataset_info.index:
            if index in bl0 + bl1:
                df_dataset_info.loc[index, 'Black_List'] = True

        bl_all = df_dataset_info[df_dataset_info['Black_List']].index.to_list()

        import json

        code_file = os.path.split(__file__)[-1]  # 获取代码文件名 *.py
        json_file = code_file[:-3] + '_BL.json'
        with open(json_file, 'w') as f:
            json.dump(bl_all, f)

        df_dataset_info.to_csv(self.info_file)
        self.logging_dataset_info()


# 添加配置文件
eeg_config = {'duration': 1.5,
              'stride': 1.0,
              'CF': 1,

              'samples_balance': False,
              'over_sampling': False,

              'feature_dim': 512,
              'fc_dim': 64,

              'data_normalize': False,  # ????
              'data_transform': True,
              'T': 1.0,                 # transformer-like(1/dk)

              'epochs': 200,
              'batch_size': 128,
              'lr_init': 1.0e-3,
              'epoch_decay_start': 80,
              'weight_decay': 1.0e-4,

              'net': EEGNet_PT452_FC1,
              'dataset_train': DatasetCoTeachingArrayBalance,
              'dataset_predict': DatasetCoTeachingArray,

              'init_epoch': 5,     # warm-up
              'forget_rate': 0.2,
              'num_gradual': 10,
              'exponent': 1,
              'loss_fn': loss_coteaching_plus_m5,

              'Note': 'Co-teaching Plus for qualify with SimSiam feature.'
              }


#keyword = 'DC'
keyword = ('DC', 'TD') 

black_list_file = ''
#features_path = '../../SimSiam/Features/EEGLab_SimSiam_new_1.5_1.0/'
features_path = 'C:/Users/user/pythonproject/REAL/SimSiam/Features/EEGLab_SimSiam_new_1.5_1.0'

if __name__ == '__main__':
    # 1. Input Data
    eeg = EEGLabLOSOFeature()
    eeg.register_logging()
    eeg.collect_dataset_info_with_black_list()
    eeg.make_epochs_sn_df()
    eeg.load_feature_data_to_array()

    # 2. Model Process
    eeg.fit_method = eeg.train_co_teaching_plus_adam
    for g in range(5):
        eeg.group = g
        logging.info('g: %d' % (g,))
        eeg.make_epochs_sn_df()
        eeg.evaluate_self_feature()
        eeg.save_epochs_info_as(g)

    # 3. Output Result
    eeg.vote_and_save()
    eeg.summary()

"""
CSV文件过大的解决方法:
https://blog.csdn.net/weixin_45727931/article/details/117351775
"""
