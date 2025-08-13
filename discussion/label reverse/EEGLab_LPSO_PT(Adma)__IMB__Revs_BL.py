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

from EEGLab.EEGLab_Logging import EEGLabLogging
from EEGLab.EEGLab_Transform import *

from SimSiamLib.LearningRate import *
from SimSiamLib.EEGLab_Tools import *

from Co_teachingLib.Models import EEGNet_PT452_FC1
from Co_teachingLib.Dataset import DatasetCoTeachingArray

import logging

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


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


class EEGLabLOSOFeaturePytorch(object):
    """
    self.channel_factor = 1

    对train-set & val-set均使用over-sampling.
    建议:
      调用cnn_model.train_on_batch()时, 使用class_weight.
      调用cnn_model.evaluate()时, 使用sample_weight!

    使用独立的evaluate()检验训练集的性能, 放弃误差较大的train_on_batch()返回值.
    """

    def __init__(self):
        super().__init__()

        self.duration = eeg_config['duration']
        self.stride = eeg_config['stride']
        self.overlap = self.duration - self.stride
        self.channel_factor = eeg_config['CF']

        self.eeg_config = eeg_config

        self.stride_epochs = 1.0
        self.n_blocks = 1

        self.samples = int(self.duration * 250.0 + 1)
        self.noise_rate = 0.2  # DisturbLabel

        self.split_random_state = 0  # 随机分割数据集的种子, 被log收集. user.split_random_state
        self.sn_kfold = 0  # KFold的序号, 被log收集. user.sn_kfold

        self.step = None
        self.stage = None
        self.round = None

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
        self.bone_file_name = '_'.join([code_file_name, class_all]) + '_' + keyword

        # info文件名
        self.info_file = self.bone_file_name + '_info.csv'
        self.info_bak_file = self.bone_file_name + '_info_bak.csv'
        self.epochs_file = self.bone_file_name + '_epochs.csv'

        self.evaluate_file = self.bone_file_name + '_evaluate.csv'

        # 筛选所用文件
        self.epochs_offset_sel_file = self.bone_file_name + '_epochs_offset_sel.csv'
        self.examples_offset_sel_file = self.bone_file_name + '_examples_offset_sel.csv'

        # 测试建模文件
        self.epochs_offset_group_file = self.bone_file_name + '_epochs_offset_group.csv'
        self.examples_offset_group_file = self.bone_file_name + '_examples_offset_group.csv'



        # self.examples_sn_file = self.bone_file_name + '_examples_sn.csv'
        # self.info_bak_file = self.bone_file_name + '_info_bak.csv'

        # dataFrame数据结构
        self.df_dataset_info = None
        self.df_epochs_info = None

        # 实例化EEGLabLogging类, 模型与历史文件管理
        self.log = EEGLabLogging(user=self)

        # 声明. 避免后续赋值时出现的"下划波浪线"
        self.x_train, self.x_val, self.x_test = None, None, None
        self.y_train, self.y_val, self.y_test = None, None, None

        self.df_train, self.df_val, self.df_test = None, None, None

        self.dataset_dict = None
        self.n_splits = 5

        # 学习率调度
        # self.schedule = schedule_step_closure(lr_init=0.0001, epochs=200, interval=50)
        # self.schedule = schedule_none_closure(lr_init=0.0002)
        # self.schedule = schedule_step_closure(lr_init=0.0001, epochs=100, interval=25)
        self.schedule = schedule_step_closure(lr_init=0.0002, epochs=100, interval=25)

        self.fit_method = None

        self.group = 0

    def register_logging(self):
        logging.basicConfig(filename=self.bone_file_name + '.log', level=logging.INFO,
                            format='%(asctime)s %(message)s')

        logging.info('duration: %.2f' % (self.duration,))
        logging.info('stride: %.2f' % (self.stride,))
        logging.info('channel_factor: %d' % (self.channel_factor,))

        logging.info('samples_balance: %s' % (eeg_config['samples_balance'],))
        # logging.info('test_set_shuffle: %s' % (eeg_config['test_set_shuffle'], ))

        # logging.info('noise_rate: %.2f' % (eeg_config['noise_rate'], ))
        logging.info('lr_init: %.2e' % (eeg_config['lr_init'],))

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
            n_times = row['n_times']

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

    def n_times_to_epochs_offset_group(self, n_times, shuffle=False):
        """
        按设置参数分割fif至epochs group.
        :param n_times: int
        :param shuffle:
        :return: x_all
        """
        stride = int(self.stride_epochs * 250.)
        duration = int(self.duration * 250.0 + 1)
        n_epochs = (n_times - duration) // stride + 1

        offset_sel = np.arange(n_epochs) * stride

        if shuffle:
            gen_array_list = [offset_sel.reshape(-1, 1)]
            for _ in range(self.channel_factor - 1):
                c_n = (gen_array_list[-1].copy())[-1::-1, :]
                gen_array_list.append(c_n)
            return np.concatenate(gen_array_list, axis=1)
        else:
            offset_sel = offset_sel.reshape(-1, 1)
            offset_sel = np.repeat(offset_sel, self.channel_factor, axis=1)
            return offset_sel

    def make_epochs_offset_group_df(self, shuffle=False):
        """
        :return:
        """
        print('>>', 'make_epochs_offset_group_df()')
        mkdir(epochs_path)

        df_dataset_info = pd.read_csv(self.info_file)

        df_dict = {'file': [], 'upper_limit': [], 'fif_idx': [], 'ID': [], 'sn_str': [], 'label': []}

        # 逐行处理df_dataset_info
        for idx in range(len(df_dataset_info)):
            row = df_dataset_info.iloc[idx]  # 读一行
            file_path = row['file'].strip()
            label = row['label']
            # sub_dataset = row['dataset']
            n_times = row['n_times']

            offset_sel = self.n_times_to_epochs_offset_group(n_times, shuffle=shuffle)
            # print(offset_sel.shape)

            # 以fif文件名基础, 生成的epochs独立编排序号
            base_file = os.path.split(file_path)[-1]

            # 组成example的epochs序号存入df
            for m in range(offset_sel.shape[0]):
                example_sn_str = [str(idx) for idx in offset_sel[m]]

                sn_str = ','.join(example_sn_str)
                df_dict['sn_str'].append(sn_str)

                epoch_file = base_file + '_xy'
                epoch_path = os.path.join(epochs_path, epoch_file)
                df_dict['file'].append(epoch_path)

            df_dict['label'] += [label] * offset_sel.shape[0]
            df_dict['fif_idx'] += [idx] * offset_sel.shape[0]
            df_dict['upper_limit'] += [offset_sel.shape[0]] * offset_sel.shape[0]
            df_dict['ID'] += list(range(offset_sel.shape[0]))

        df_examples_info = pd.DataFrame(df_dict, index=[n for n in range(len(df_dict['file']))])
        df_examples_info['step'] = 0
        df_examples_info.to_csv(self.epochs_offset_group_file)

    ######################################################

    def n_times_to_examples_offset_group(self, n_times, coef=1.0):
        """

        :param n_times: int
        :param coef: float
        :return: x_all
        """
        stride = int(self.stride * 250.0 / coef)
        duration = int(self.duration * 250.0 + 1)
        n_epochs = (n_times - duration) // stride + 1

        examples_offset = np.arange(n_epochs) * stride

        # gen_array_list = []
        # for _ in range(self.channel_factor):
        #     c_n = np.random.permutation(examples_offset).reshape(-1, 1)
        #     gen_array_list.append(c_n)
        # return np.concatenate(gen_array_list, axis=1)

        gen_array_list = [examples_offset.reshape(-1, 1)]
        for _ in range(self.channel_factor - 1):
            c_n = (gen_array_list[-1].copy())[-1::-1, :]  # 反褶
            gen_array_list.append(c_n)
        return np.concatenate(gen_array_list, axis=1)

    def make_examples_offset_group_df(self, samples_balance=False):
        """
        根据_info.csv文件, 分割fif为epochs并组合为examples, 索引信息存至_examples.csv.
        视samples_balance决定是否over-sampling.
        如需over-sampling,将自动计算over-sampling coefficient
        :param samples_balance: bool
        :return:
        """
        print('>>', 'make_examples_offset_group_df()')
        # mkdir(epochs_path)

        df_dataset_info = pd.read_csv(self.info_file)

        # 计算train样本不均衡率
        over_sampling_coef = {0: 1.0, 1: 1.0}
        if samples_balance:
            df = df_dataset_info  # [df_dataset_info['dataset'] != 'unqualified']
            df_0 = df[df['label'] == 0]
            df_1 = df[df['label'] == 1]
            if len(df_1) > len(df_0):
                over_sampling_coef[0] = len(df_1) / len(df_0)
            else:
                over_sampling_coef[1] = len(df_0) / len(df_1)
        print('over_sampling_coeff', over_sampling_coef)

        df_dict = {'file': [], 'upper_limit': [], 'fif_idx': [], 'ID': [], 'sn_str': [], 'label': []}

        # 逐行处理df_dataset_info
        for idx in range(len(df_dataset_info)):
            row = df_dataset_info.iloc[idx]  # 读一行
            file_path = row['file'].strip()
            label = row['label']
            n_times = row['n_times']

            offset_sel = self.n_times_to_examples_offset_group(n_times, coef=over_sampling_coef[label])

            # 以fif文件名基础, 生成的epochs独立编排序号
            base_file = os.path.split(file_path)[-1]

            # 组成example的epochs序号存入df
            for m in range(offset_sel.shape[0]):
                example_sn_str = [str(idx) for idx in offset_sel[m]]

                sn_str = ','.join(example_sn_str)
                df_dict['sn_str'].append(sn_str)

                epoch_file = base_file + '_xy'
                epoch_path = os.path.join(epochs_path, epoch_file)
                df_dict['file'].append(epoch_path)

            df_dict['label'] += [label] * offset_sel.shape[0]
            df_dict['fif_idx'] += [idx] * offset_sel.shape[0]
            df_dict['upper_limit'] += [offset_sel.shape[0]] * offset_sel.shape[0]
            df_dict['ID'] += list(range(offset_sel.shape[0]))
            # df_dict['dataset'] += [sub_dataset]*sn_sel.shape[0]

        df_examples_info = pd.DataFrame(df_dict, index=[n for n in range(len(df_dict['file']))])
        df_examples_info['step'] = 0

        # for CL
        df_examples_info['Pred'] = df_examples_info['label']
        df_examples_info['Prob0'] = 0.0
        df_examples_info['Prob1'] = 0.0
        df_examples_info['SC'] = 1.0
        df_examples_info['y_0'] = df_examples_info['label'] == 0
        df_examples_info['y_1'] = df_examples_info['label'] == 1
        df_examples_info['y_star'] = df_examples_info['label']

        df_examples_info.to_csv(self.examples_offset_group_file)

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
            print(file_path)

            base_file = os.path.split(file_path)[-1]
            features_file_path = os.path.join(features_path, base_file + '.pkl')
            with open(features_file_path, 'rb') as f:
                x_all = pickle.load(f)

                if eeg_config['data_transform']:
                    x_all = SelfAttention()(x_all)

            if eeg_config['data_normalize']:
                x_all = x_all / np.linalg.norm(x_all, axis=1, keepdims=True)

            self.dataset_dict[idx] = x_all

        # total = 0
        # for key in self.dataset_dict:
        #     total += self.dataset_dict[key].shape[2]
        # print(total)

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

    def class_weight_df_PT(self, df):
        """
        计算不同类别examples的不均衡度, 不同的方法对acc会有一定影响.
        :param df:
        :return: class_weight dict
        """
        label_1 = df['label'].sum()
        label_0 = len(df) - label_1
        # labels = [(df['label'] == n).sum() for n in range(2)]
        print(label_0, label_1)

        class_weight = np.array([2*label_1 / len(df), 2*label_0 / len(df)], dtype='float32')
        # class_weight = {0: label_1 / len(df), 1: label_0 / len(df)}
        print(class_weight)

        return torch.from_numpy(class_weight)

    def df_dataset_to_array_with_offset_ds(self, df_dataset, label_noise=0.0):
        """
        借用DataSequence
        依据DF形式的数据集, 导出np.array形式的常规数据集.
        :param df_dataset: df of examples
        :return:
        """

        dataset_array_list = []
        for x, _ in DataSequenceDictIndex(df_dataset,
                                          batch_size=1024,
                                          shuffle=False,
                                          dataset_dict=self.dataset_dict,
                                          label_noise=label_noise,
                                          samples=self.samples,
                                          axis=1):
            # (1024, 30*2, 251, 1)
            dataset_array_list.append(x)

        return np.concatenate(dataset_array_list, axis=0).astype('float32')  # (N, 30*2, 251, 1)

    ########################################################

    def fit_offset(self, epochs=None, batch_size=None):
        """
        数据集一次性加载至GPU, 适合高端显卡! 速度快.
        """
        from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

        # select model
        cnn_model = EEGNet_v452_FC(nb_classes=self.n_classes, input_dim=512)
        print(cnn_model.name)

        # 自动生成模型文件名/训练历史文件名
        model_file, history_file = self.log.gen_model_history_file_name_from_log_file(cnn_model.name)
        class_weight_training = self.class_weight_df(self.df_train)

        check_pointer = ModelCheckpoint(monitor=eeg_config['check_point_monitor'],  # 'val_accuracy',
                                        filepath=model_file,
                                        verbose=1,
                                        save_best_only=True)

        early_stopping = EarlyStopping(monitor='val_accuracy', patience=50)

        # 导入配置数据
        epochs = eeg_config['epochs']
        batch_size = eeg_config['batch_size']

        # 自定义学习率调整
        # schedule_step = schedule_cosine_decay_closure(lr_init=eeg_config['lr_init'], epochs=100)
        schedule_step = schedule_cosine_decay_warmup_closure(lr_init=eeg_config['lr_init'], epochs=epochs,
                                                             warmup=epochs / 10)
        # schedule_step = schedule_step_closure(lr_init=eeg_config['lr_init'], epochs=epochs, interval=25)
        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(schedule_step, verbose=0)

        # 生成 array 格式数据集
        array_train = self.df_dataset_to_array_with_offset_ds(self.df_train)
        array_val = self.df_dataset_to_array_with_offset_ds(self.df_val)

        # 重大修改后的 fit
        fit_history = cnn_model.fit(x=array_train,
                                    y=self.df_train['label'].to_numpy(),
                                    batch_size=batch_size,
                                    epochs=epochs,
                                    validation_data=(array_val, self.df_val['label'].to_numpy()),

                                    # steps_per_epoch=self.df_train.shape[0] // batch_size,
                                    # validation_steps=self.df_val.shape[0] // batch_size,

                                    class_weight=class_weight_training,
                                    callbacks=[check_pointer, lr_scheduler],  # early_stopping],
                                    verbose=2,
                                    initial_epoch=0)

        # # Load the best model
        # from tensorflow.keras.models import load_model
        #
        # custom_objects = {
        #     "Slice": Slice
        # }
        # best_model = load_model(model_file, custom_objects=custom_objects)
        #
        # array_test = self.df_dataset_to_array_with_offset_ds(self.df_test)
        # test_loss, test_acc = best_model.evaluate(x=array_test,
        #                                           y=self.df_test['label'].to_numpy(),
        #                                           batch_size=batch_size,
        #                                           verbose=0)
        #
        # fit_history.history['test_accuracy'] = test_acc

        # save the latest model for transform
        self.log.save_final_model_history(cnn_model, pd.DataFrame(fit_history.history))

        # 不知哪个好使:)
        del cnn_model
        K.clear_session()

    def fit_phases(self, epochs=None, batch_size=None):
        """
        数据集一次性加载至GPU, 适合高端显卡! 速度快.
        """
        from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

        # 导入配置数据
        epochs = eeg_config['epochs']
        batch_size = eeg_config['batch_size']
        epochs_phase_1 = eeg_config['epochs_phase_1']
        warmup = eeg_config['warmup']
        l2_weight = eeg_config['l2_weight']
        input_dim = eeg_config['feature_dim']

        # select model
        net = eeg_config['net']
        cnn_model = net(nb_classes=self.n_classes, input_dim=input_dim, l2_weight=l2_weight)
        print(cnn_model.name)

        # 自动生成模型文件名/训练历史文件名
        model_file = self.log.open(cnn_model.name)

        # 设置 Checkpoint & EarlyStopping
        check_pointer = ModelCheckpoint(monitor=eeg_config['check_point_monitor'],  # 'val_accuracy',
                                        filepath=model_file,
                                        verbose=1,
                                        save_best_only=True)

        early_stopping = EarlyStopping(monitor='val_accuracy', patience=epochs // 2)

        # 自定义学习率调整
        # schedule_step = schedule_cosine_decay_closure(lr_init=eeg_config['lr_init'], epochs=100)
        schedule = schedule_cosine_decay_warmup_closure(lr_init=eeg_config['lr_init'], epochs=epochs, warmup=warmup)
        # schedule_step = schedule_step_closure(lr_init=eeg_config['lr_init'], epochs=epochs, interval=25)
        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(schedule, verbose=0)

        # 生成 array 格式数据集
        array_train = self.df_dataset_to_array_with_offset_ds(self.df_train, label_noise=eeg_config['label_noise'])
        array_val = self.df_dataset_to_array_with_offset_ds(self.df_val)
        array_test = self.df_dataset_to_array_with_offset_ds(self.df_test)

        class_weight_training = self.class_weight_df(self.df_train)

        # 重大修改后的 fit
        fit_history1 = cnn_model.fit(x=array_train,
                                     y=self.df_train['label'].to_numpy(),
                                     batch_size=batch_size,  # ****
                                     epochs=epochs_phase_1,
                                     validation_data=(array_val, self.df_val['label'].to_numpy()),
                                     class_weight=class_weight_training,
                                     callbacks=[lr_scheduler],
                                     verbose=2,
                                     initial_epoch=0)

        # 重大修改后的 fit
        fit_history2 = cnn_model.fit(x=array_train,
                                     y=self.df_train['label'].to_numpy(),
                                     batch_size=batch_size,  # ****
                                     epochs=epochs,
                                     validation_data=(array_val, self.df_val['label'].to_numpy()),
                                     class_weight=class_weight_training,
                                     callbacks=[check_pointer, lr_scheduler, early_stopping],
                                     verbose=2,
                                     initial_epoch=epochs_phase_1)

        # 合并历史
        history = {}
        for key in fit_history1.history:
            history[key] = fit_history1.history[key] + fit_history2.history[key]

        self.log.monitor.update_all(history)

        # save
        self.log.close()

        # 不知哪个好使:)
        del cnn_model
        K.clear_session()

    def fit_on_batch_phases(self, epochs=100, batch_size=512, channel_factor=1):
        """
        train_on_batch():
            label_noise=self.noise_rate
            class_weight=class_weight_training
        逐mini—batch训练, 并逐回合评估loss与accuracy.
        可以监视训练过程, 保存最优模型, 必要时提前终止.
        资源占用少, 自主可控.
        """
        import time
        import tensorflow.keras.backend as K

        # 导入配置数据
        epochs = eeg_config['epochs']
        batch_size = eeg_config['batch_size']
        epochs_phase_1 = eeg_config['epochs_phase_1']
        warmup = eeg_config['warmup']
        l2_weight = eeg_config['l2_weight']
        input_dim = eeg_config['feature_dim']

        # select model
        net = eeg_config['net']
        cnn_model = net(nb_classes=self.n_classes, input_dim=input_dim, l2_weight=l2_weight)
        print(cnn_model.name)

        # 自动生成模型文件名/训练历史文件名
        model_file = self.log.open(cnn_model.name)

        # 生成 array 格式数据集
        array_train = self.df_dataset_to_array_with_offset_ds(self.df_train)
        array_val = self.df_dataset_to_array_with_offset_ds(self.df_val)

        steps = len(DataSequenceArrayIndex(array_train, None, batch_size=batch_size))
        class_weight_training = self.class_weight_df(self.df_train)

        self.schedule = schedule_cosine_decay_warmup_closure(lr_init=eeg_config['lr_init'], epochs=epochs, warmup=warmup)

        # Early Stopping
        patience = epochs // 2
        early_stopping_cnt = 0

        # 逐回合训练
        val_acc_best = 0.0
        val_loss_best = 10.0
        for epoch in range(epochs):
            print("\nEpoch {0}/{1}".format(str(epoch + 1), str(epochs)))

            # 0. set learning rate every epoch
            lr = self.schedule(epoch)
            K.set_value(cnn_model.optimizer.lr, lr)  # set lr
            lr = float(K.get_value(cnn_model.optimizer.lr))  # get lr

            t0 = time.time()

            # 1. 使用数据生成器生成batches, 对每一个batch进行训练
            for m, (x, y) in enumerate(DataSequenceArrayIndex(array_train,
                                                              self.df_train['label'].to_numpy(),
                                                              batch_size=batch_size,
                                                              shuffle=True,
                                                              samples=self.samples,
                                                              label_noise=eeg_config['label_noise'],
                                                              axis=1),
                                       1):

                loss, acc = cnn_model.train_on_batch(x, y,
                                                     class_weight=class_weight_training,
                                                     reset_metrics=False)

                # keras-like display
                end = '\n' * (m // steps)
                print("\r" + "{0}/{1} - loss: {2:.4f} - accuracy: {3:.4f}".format(str(m).rjust(len(str(steps))),
                                                                                  str(steps), loss, acc), end=end)

            # 3. 一个epoch结束, 评估model在验证集上的性能
            val_loss, val_acc = cnn_model.evaluate(DataSequenceArrayIndex(array_val,
                                                                          self.df_val['label'].to_numpy(),
                                                                          batch_size=batch_size * 2,
                                                                          samples=self.samples,
                                                                          dataset_dict=self.dataset_dict,
                                                                          axis=1),
                                                   verbose=0)
            t1 = time.time()
            t = int(t1 - t0)

            self.log.monitor.update(accuracy=acc, loss=loss, val_accuracy=val_acc, val_loss=val_loss, lr=lr)
            self.log.monitor.report()

            # 4. save the best acc model
            if val_acc >= val_acc_best and epoch > epochs_phase_1:
                cnn_model.save(model_file)
                val_acc_best = val_acc
                early_stopping_cnt = 0

            # 5. save the best loss model
            if val_loss <= val_loss_best and epoch > epochs_phase_1:
                cnn_model.save(model_file[:-3] + '_loss.h5')
                val_loss_best = val_loss

            # # 6. save the latest model for transform every epoch
            # self.log.save_final_model_history(cnn_model, pd.DataFrame(monitor.history))

            # Early Stopping
            early_stopping_cnt += 1
            if early_stopping_cnt > patience:
                break

        # save
        self.log.close()

        del cnn_model
        K.clear_session()

    def fit_on_batch(self, epochs=100, batch_size=512, channel_factor=1):
        """
        train_on_batch():
            label_noise=self.noise_rate
            class_weight=class_weight_training
        逐mini—batch训练, 并逐回合评估loss与accuracy.
        可以监视训练过程, 保存最优模型, 必要时提前终止.
        资源占用少, 自主可控.
        """
        import time
        import tensorflow.keras.backend as K

        # select model
        # cnn_model = EEGNet_v452(nb_classes=self.n_classes,
        #                         samples=self.samples,
        #                         channels=channel_factor)

        cnn_model = EEGNet_v452_II(nb_classes=self.n_classes,
                                   samples=self.samples,
                                   channels=channel_factor)
        print(cnn_model.name)

        # 自动生成模型文件名/训练历史文件名
        model_file, history_file = self.log.gen_model_history_file_name_from_log_file(cnn_model.name)

        steps = len(DataSequenceDictOffset(self.df_train, batch_size=batch_size))
        class_weight_training = self.class_weight_df(self.df_train)

        self.schedule = schedule_step_closure(lr_init=eeg_config['lr_init'], epochs=100, interval=25)

        # Monitor: 自定义的监测记录class
        monitor = Monitor(epochs, steps, model_file)

        # Early Stopping
        patience = 50
        early_stopping_cnt = 0

        # 逐回合训练
        val_acc_best = 0.0
        val_loss_best = 10.0
        test_loss = 0.0
        test_acc = 0.0
        for epoch in range(epochs):
            print("\nEpoch {0}/{1}".format(str(epoch + 1), str(epochs)))

            # 0. set learning rate every epoch
            lr = self.schedule(epoch)
            K.set_value(cnn_model.optimizer.lr, lr)  # set lr
            lr = float(K.get_value(cnn_model.optimizer.lr))  # get lr

            t0 = time.time()

            # 1. 使用数据生成器生成batches, 对每一个batch进行训练
            for m, (x, y) in enumerate(DataSequenceDictOffset(self.df_train,
                                                              batch_size=batch_size,
                                                              shuffle=True,
                                                              dataset_dict=self.dataset_dict,
                                                              # method='shuffle',
                                                              samples=self.samples,
                                                              label_noise=self.noise_rate, axis=1),
                                       1):
                loss, acc = cnn_model.train_on_batch(x, y,
                                                     class_weight=class_weight_training,
                                                     reset_metrics=False)

                # keras-like display
                end = '\n' * (m // steps)
                print("\r" + "{0}/{1} - loss: {2:.4f} - accuracy: {3:.4f}".format(str(m).rjust(len(str(steps))),
                                                                                  str(steps), loss, acc), end=end)

            # 2. 一个epoch结束, 评估model在测试集上的性能. train_on_batch的返回值误差较大!
            loss, acc = cnn_model.evaluate(DataSequenceDictOffset(self.df_train,
                                                                  batch_size=batch_size * 2,
                                                                  samples=self.samples,
                                                                  dataset_dict=self.dataset_dict, axis=1),
                                           verbose=0)
            print(loss, acc)

            # 3. 一个epoch结束, 评估model在验证集上的性能
            val_loss, val_acc = cnn_model.evaluate(DataSequenceDictOffset(self.df_val,
                                                                          batch_size=batch_size * 2,
                                                                          samples=self.samples,
                                                                          dataset_dict=self.dataset_dict, axis=1),
                                                   verbose=0)
            t1 = time.time()
            t = int(t1 - t0)

            # 4. save the best acc model
            if val_acc >= val_acc_best:
                cnn_model.save(model_file)
                val_acc_best = val_acc
                early_stopping_cnt = 0  # Early Stopping

                # # 4.1 评估测试集性能
                # test_loss, test_acc = cnn_model.evaluate(DataSequenceDictOffset(self.df_test,
                #                                                                 batch_size=batch_size,
                #                                                                 dataset_dict=self.dataset_dict, axis=1),
                #                                          verbose=0)
                # print("test_loss: {0:.4f} - test_acc: {1:.4f}".format(test_loss, test_acc))

            monitor.rec_metrics(loss, acc, val_loss, val_acc, lr, test_loss, test_acc)
            monitor.rec_other(epoch, t)
            monitor.report()

            # 5. save the best loss model
            if val_loss <= val_loss_best:
                cnn_model.save(model_file[:-3] + '_loss.h5')
                val_loss_best = val_loss

            # 6. save the latest model for transform every epoch
            self.log.save_final_model_history(cnn_model, pd.DataFrame(monitor.history))

            # Early Stopping
            early_stopping_cnt += 1
            if early_stopping_cnt > patience:
                break

        # 不知哪个好使:)
        del cnn_model
        K.clear_session()

    ########################################################

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

    def df_fif_to_df_examples(self, df_fif):
        """
        逐行读入df, 根据df的file, 在_examples.csv中选取对应的examples, 组成数据集.
        :param df_fif:
        :return:
        """
        df_examples_all = pd.read_csv(self.examples_offset_group_file, index_col=0, dtype={'sn_str': str})

        df_list = []
        for index in df_fif.index:
            # examples of fif
            df_examples_of_fif = df_examples_all[df_examples_all['fif_idx'] == index].copy()

            # # CL合格的
            # df_cl = df_examples_of_fif[df_examples_of_fif['label'] == df_examples_of_fif['y_star']]
            df_list.append(df_examples_of_fif)

        df = pd.concat(df_list, axis=0)
        return df

    def df_fif_to_df_examples_CL(self, df_fif):
        """
        逐行读入df, 根据df的file, 在_examples.csv中选取对应的examples, 组成数据集.
        :param df_fif:
        :return:
        """
        df_examples_all = pd.read_csv(self.examples_offset_group_file, index_col=0, dtype={'sn_str': str})

        df_list = []
        for index in df_fif.index:
            # examples of fif
            df_examples_of_fif = df_examples_all[df_examples_all['fif_idx'] == index].copy()

            # CL合格的
            df_cl = df_examples_of_fif[df_examples_of_fif['label'] == df_examples_of_fif['y_star']]
            df_list.append(df_cl)

        df = pd.concat(df_list, axis=0)
        return df

    def df_fif_to_df_examples_offset_group(self, df_fif):
        """
        逐行读入df, 根据df的file, 在_examples.csv中选取对应的examples, 组成数据集.
        :param df_fif:
        :return:
        """
        df_examples = pd.read_csv(self.examples_offset_group_file, index_col=0, dtype={'sn_str': str})
        df_list = []

        for idx in range(len(df_fif)):
            row = df_fif.iloc[idx]  # 读一行
            file_path = row['file'].strip()
            file = os.path.split(file_path)[-1]

            # 对应的examples
            df_dataset_m = df_examples[df_examples['file'].str.contains(file, regex=False)].copy()
            # print(df_dataset_m)
            df_list.append(df_dataset_m)

        df_dataset = pd.concat(df_list, axis=0)
        return df_dataset

    def df_fif_to_df_epochs_offset_group(self, df_fif):
        """
        逐行读入df, 根据df的file, 在_examples.csv中选取对应的examples, 组成数据集.
        :param df_fif:
        :return:
        """
        df_epochs = pd.read_csv(self.epochs_offset_group_file, index_col=0, dtype={'sn_str': str})
        df_list = []

        for idx in range(len(df_fif)):
            row = df_fif.iloc[idx]  # 读一行
            file_path = row['file'].strip()
            file = os.path.split(file_path)[-1]

            # 对应的examples
            df_dataset_m = df_epochs[df_epochs['file'].str.contains(file, regex=False)].copy()
            # df_dataset_m = df_dataset_m[df_dataset_m['p'] < median].copy()
            # print(df_dataset_m)
            df_list.append(df_dataset_m)

        df_dataset = pd.concat(df_list, axis=0)
        return df_dataset

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

    def predict_examples_CL(self, df_examples_to_predict, model_indices=None, n_splits=5):
        """
        评估模型在指定数据集上的性能.
        :param df_examples_to_predict:
        :param model_indices: list | int
        :param n_splits: 5
        :return:
        """
        from tensorflow.keras.models import load_model

        df_log = pd.read_csv(self.log.log_file)

        y_probs = 0
        for k in range(n_splits):
            if isinstance(model_indices, int):
                model_file = df_log['h5_csv'].iloc[model_indices - k] + '.h5'
            else:
                if k >= len(model_indices):
                    break
                model_file = df_log['h5_csv'].iloc[model_indices[k]] + '.h5'
            print(model_file)

            # Load model
            custom_objects = {
                "Slice": Slice
                # "SliceEpochs": SliceEpochs,
                # "MaxPoolingEEGChannel": MaxPoolingEEGChannel
            }
            cnn_model = load_model(model_file, custom_objects=custom_objects)

            ds = DataSequenceDictOffset(df_examples_to_predict, batch_size=32,
                                        dataset_dict=self.dataset_dict,
                                        shuffle=False,
                                        axis=1)

            """
            WARNING:tensorflow:6 out of the last 11 calls to <function Model.make_predict_function
            model.predict(X) -> model(X)
            """
            y_prob = cnn_model.predict(ds, steps=len(ds))
            y_probs += y_prob

        y_prob = y_probs / n_splits
        # y_pred = np.argmax(y_prob, axis=1)

        self.label_examples_CL(df_examples_to_predict, y_prob)

    def label_examples_CL(self, df_examples_to_label, y_probs):
        """
        标注相关epochs的预测输出值. (0, 1)
        :param df_examples_to_label: 使用其索引去标注epochs_file中对应的epochs
        :param y_probs:
        :return:
        """

        y_preds = np.argmax(y_probs, axis=1)

        df_examples = pd.read_csv(self.examples_offset_group_file, index_col=0)

        for n, idx in enumerate(df_examples_to_label.index):
            label = df_examples.loc[idx, 'label']
            df_examples.loc[idx, 'Pred'] = y_preds[n]

            df_examples.loc[idx, 'Prob0'] = y_probs[n, 0]
            df_examples.loc[idx, 'Prob1'] = y_probs[n, 1]
            df_examples.loc[idx, 'SC'] = y_probs[n, label]

        df_examples.to_csv(self.examples_offset_group_file)

    ########################################################

    def predict_epochs_PT(self, df_epochs_to_predict, model_indices=None):
        """
        评估模型在指定数据集上的性能.
        :param df_epochs_to_predict: DataFrame
        :param model_indices: list
        :return:
        """
        df_log = pd.read_csv(self.log.log_file)
        y_probs = np.zeros((len(df_epochs_to_predict), 2), dtype='float32')

        test_loader = self.build_loader(df_epochs_to_predict, batch_size=1024)

        for model_index in model_indices:
            state_dict_file = df_log['h5_csv'].iloc[model_index] + '.pt'
            print(state_dict_file)

            logging.info(state_dict_file)

            # 剔除异常模型
            if eeg_config['drop_bad_model']:
                val_acc = df_log['val_acc'].iloc[model_index]
                val_loss = df_log['val_loss'].iloc[model_index]

                if val_acc < eeg_config['val_acc_th']:
                    logging.info('drop!')
                    continue

                # if val_loss > eeg_config['val_loss_th'] and val_acc < eeg_config['val_acc_th']:
                #     logging.info('drop!')
                #     continue

            net = eeg_config['net']
            model = net(self.n_classes, eeg_config['feature_dim'])
            state_dict = torch.load(state_dict_file)
            model.load_state_dict(state_dict, strict=False)
            model.cuda(device)

            y_prob = self.probs_pytorch(model, test_loader)
            y_probs += y_prob

        # y_prob = y_probs / len(model_indices)
        y_prob = y_probs / y_probs.sum(axis=1, keepdims=True)

        self.label_epochs_PT(df_epochs_to_predict, y_prob)

    def label_epochs_PT(self, df_epochs_to_label, y_probs):
        """
        标注相关 epochs 的预测输出值. (0, 1)
        :param df_epochs_to_label: 使用其索引去标注 epochs_file 中对应的 epochs
        :param y_probs:
        :return:
        """

        y_preds = np.argmax(y_probs, axis=1)

        # logging
        acc = (y_preds == df_epochs_to_label['label'].to_numpy()).mean()
        logging.info('Epochs Acc: %.4f\n' % (acc,))

        df_epochs = pd.read_csv(self.epochs_file, index_col=0)

        for n, idx in enumerate(df_epochs_to_label.index):
            label = df_epochs.loc[idx, 'label']
            df_epochs.loc[idx, 'Pred'] = y_preds[n]

            df_epochs.loc[idx, 'Prob0'] = y_probs[n, 0]
            df_epochs.loc[idx, 'Prob1'] = y_probs[n, 1]
            df_epochs.loc[idx, 'SC'] = y_probs[n, label]

        df_epochs.to_csv(self.epochs_file)

    ########################################################

    def load_model_with_log_index(self, log_index=0):
        from tensorflow.keras.models import load_model

        df_log = pd.read_csv(self.log.log_file)
        model_file = df_log['h5_csv'].iloc[log_index] + '.h5'
        print(model_file)

        # Load model
        custom_objects = {
            "Slice": Slice
            # "SliceEpochs": SliceEpochs,
            # "MaxPoolingEEGChannel": MaxPoolingEEGChannel
        }

        model = load_model(model_file, custom_objects=custom_objects)

        return model

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
            # plt.title(dataset)
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

    def predict_epochs(self, df_epochs_to_predict, model_indices=None, n_splits=5):
        """
        评估模型在指定数据集上的性能.
        :param df_epochs_to_predict:
        :param model_indices: list | int
        :param n_splits: 5
        :return:
        """
        from tensorflow.keras.models import load_model

        df_log = pd.read_csv(self.log.log_file)

        y_probs = 0
        for k in range(n_splits):
            if isinstance(model_indices, int):
                model_file = df_log['h5_csv'].iloc[model_indices - k] + '.h5'
            else:
                if k >= len(model_indices):
                    break
                model_file = df_log['h5_csv'].iloc[model_indices[k]] + '.h5'
            print(model_file)

            # Load model
            custom_objects = {
                "Slice": Slice
                # "SliceEpochs": SliceEpochs,
                # "MaxPoolingEEGChannel": MaxPoolingEEGChannel
            }
            cnn_model = load_model(model_file, custom_objects=custom_objects)

            ds = DataSequenceDictOffset(df_epochs_to_predict, batch_size=256,
                                        dataset_dict=self.dataset_dict,
                                        shuffle=False,
                                        axis=1)

            y_prob = cnn_model.predict(ds, steps=len(ds))
            y_probs += y_prob

        y_prob = y_probs / n_splits
        # y_pred = np.argmax(y_prob, axis=1)

        self.label_epochs_sn(df_epochs_to_predict, y_prob)

    def label_epochs_sn(self, df_epochs_to_label, y_prob):
        """
        标注相关epochs的预测输出值. (0, 1)
        :param df_epochs_to_label: 使用其索引去标注epochs_file中对应的epochs
        :param y_prob:
        :return:
        """

        y_pred = np.argmax(y_prob, axis=1)

        df_epochs = pd.read_csv(self.epochs_file, index_col=0)

        cur_step = df_epochs.iloc[0]['step']
        col_pred = 'Pred' + str(cur_step)
        col_confidence = 'C' + str(cur_step)

        for idx, idx_epochs in enumerate(df_epochs_to_label.index):
            df_epochs.loc[idx_epochs, col_pred] = y_pred[idx]

            label = df_epochs.loc[idx_epochs, 'label']
            # df_epochs.loc[idx_epochs, col_confidence] = y_prob[idx, label]
            df_epochs.loc[idx_epochs, col_confidence] = y_prob[idx, y_pred[idx]]

            # df_epochs.loc[idx_epochs, 'step'] = cur_step + 1

        df_epochs.to_csv(self.epochs_file)

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

    def evaluate_all_examples_of_dataset_CL(self):
        """

        :return:
        """
        from numpy import nan

        df_examples = pd.read_csv(self.examples_offset_group_file, index_col=0)

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

        df_examples.to_csv(self.examples_offset_group_file)

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
                                          n_splits_test_set=5, random_state_test=30,
                                          P=1):
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
        # df_fif_wl = df_fif_all
        df_fif_wl = df_fif_all[(~df_fif_all['Black_List']) | (df_fif_all['label'] == 1)]
        print(df_fif_wl.shape)

        # 自定义 StratifiedKFoldDF 实例化
        skf_test = StratifiedKFoldDF(n_splits=n_splits_test_set, shuffle=True,
                                     random_state=random_state_test)

        self.split_random_state = random_state_test

        model_indices_for_bl = []

        # 1 分割白名单 fif 出 df_fif_subset & df_fif_test
        for k, (df_fif_subset, df_fif_test) in enumerate(skf_test.split(df_fif_wl)):
            logging.info('k=%d' % (k,))

            # 1.0 fifs -> epochs
            print(df_fif_test.index.to_numpy())
            self.df_test = self.df_fif_to_df_epochs(df_fif_test)

            # 1.1 对 df_fif_subset 执行KFold
            model_indices = self.train_with_k_fold_feature(df_fif_subset,
                                                           n_splits=n_splits_fold,
                                                           random_state=random_state + k * 100,
                                                           P=P)

            # 1.2 使用 n_splits_fold 个模型, 对测试集给出预测
            self.predict_epochs_PT(self.df_test, model_indices=model_indices)

            model_indices_for_bl += model_indices

        # # 2 使用 n_splits_fold*n_splits_test_set 个模型, 对BL集给出预测
        # self.predict_epochs_PT(self.df_fif_to_df_epochs(df_fif_bl), model_indices=model_indices_for_bl)

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

        # 对 df_fif_subset 执行KFold, 重复P次
        for p in range(P):

            # 自定义 StratifiedKFoldDF 实例化
            skf = StratifiedKFoldDF(n_splits=n_splits, shuffle=True, random_state=random_state + p)
            for df_fif_train, df_fif_val in skf.split(df_fif_subset):
                self.df_train = self.df_fif_to_df_epochs(df_fif_train)
                self.df_val = self.df_fif_to_df_epochs(df_fif_val)

                print(df_fif_train.index.to_numpy())
                print(df_fif_val.index.to_numpy())

                # 开始当前KFold的第k个训练
                self.fit_method()

                # 记录当前训练所得模型的索引
                model_indices.append(self.log.get_latest_model_index())

        return model_indices

    ########################################################

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
                    if not file.startswith(keyword):
                        continue

                    # file_path = file_path.ljust(75)     # 补空格, 目的在于csv文件的可读性.
                    files_list.append(file_path)
                    print(file_path)

                    df_dict['n_times'].append(raw.n_times)

            df_dict['file'] += files_list
            df_dict['class_name'] += [class_name] * len(files_list)
            df_dict['label'] += [class_labels[class_name]] * len(files_list)

        df_dataset_info = pd.DataFrame(df_dict, index=[n for n in range(len(df_dict['file']))])

        df_dataset_info['Black_List'] = False
        for index in df_dataset_info.index:
            if index in black_list:
                df_dataset_info.loc[index, 'Black_List'] = True

        df_dataset_info.to_csv(self.info_file)

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

                    raw = io.Raw(file_path, preload=False, verbose=False)
                    if len(raw.info['ch_names']) < self.eeg_channel_num:
                        continue

                    # EC | EO
                    if not file.startswith(keyword):
                        continue

                    # file_path = file_path.ljust(75)     # 补空格, 目的在于csv文件的可读性.
                    files_list.append(file_path)
                    print(file_path)

                    df_dict['n_times'].append(raw.n_times)

            df_dict['file'] += files_list
            df_dict['class_name'] += [class_name] * len(files_list)
            df_dict['label'] += [class_labels[class_name]] * len(files_list)

        df_dataset_info = pd.DataFrame(df_dict, index=[n for n in range(len(df_dict['file']))])

        import json

        with open(black_list_file, 'r') as f:
            black_list = json.load(f)

        df_dataset_info['Black_List'] = False
        for index in df_dataset_info.index:
            if index in black_list:
                df_dataset_info.loc[index, 'Black_List'] = True

        # Reverse label
        df_bl = df_dataset_info[df_dataset_info['Black_List']]
        df_dataset_info.loc[df_bl.index, 'label'] = 1 - df_dataset_info.loc[df_bl.index, 'label']
        # df_dataset_info.loc[df_bl.index, 'label'] = 0
        # df_dataset_info.loc[df_bl.index, 'label'] = 1

        df_dataset_info.to_csv(self.info_file)

        # self.df_wl = df_dataset_info[(~df_dataset_info['Black_List']) | (df_dataset_info['label' == 1])]

        df_bl = df_dataset_info[df_dataset_info['Black_List']]
        df_bl_n = df_bl[df_bl['label'] == 0]
        df_bl_p = df_bl[df_bl['label'] == 1]

        logging.info('Black List Number: %d/%d' % (len(df_bl_n), len(df_bl_p)))
        logging.info('Black List Index N: %s' % (json.dumps(df_bl_n.index.to_list()),))
        logging.info('Black List Index P: %s' % (json.dumps(df_bl_p.index.to_list()),))

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

    ########################################################

    def save_info_as(self, sn=0):
        df = pd.read_csv(self.info_file, index_col=0)
        info_new_file = self.info_file[:-4] + '_%d.csv' % (sn,)
        df.to_csv(info_new_file)

    def save_epochs_info_as(self, sn=0):
        df = pd.read_csv(self.epochs_file, index_col=0)
        info_new_file = self.epochs_file[:-4] + '_%d.csv' % (sn,)
        df.to_csv(info_new_file)

    ########################################################

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

    def evaluate_rate_of_fif(self, group_sn):
        """
        分析'最新'的 self.epochs_file, 评估fif
        :param group_sn:
        :return:
        """
        df_info = pd.read_csv(self.info_file, index_col=0)
        df_epochs = pd.read_csv(self.epochs_file, index_col=0)

        df_fif_all = pd.read_csv(self.info_file, index_col=0)
        # 不属于黑名单的数据或者数据在标签翻转之后是0
        df_fif_wl = df_fif_all[(~df_fif_all['Black_List']) | (df_fif_all['label'] == 1)]

        for index in df_info.index:
            df_epochs_of_fif = df_epochs[df_epochs['fif_idx'] == index].copy()
            df_info.loc[index, 'Rate'] = df_epochs_of_fif['Pred'].mean()

        df_info['Pred'] = df_info['Rate'] >= 0.5
        df_info.to_csv(self.info_file)  # ？？？

        def df_plot_(df_, dataset):
            from sklearn.metrics import classification_report

            plt.rcParams.update({'font.size': 14})
            plt.legend(loc='upper right')

            y_pred = df_['Pred'].to_numpy()
            y_true = df_['label'].to_numpy()

            rpt = classification_report(y_true, y_pred, output_dict=True)
            acc = rpt['accuracy']
            TPR = rpt['1']['precision']
            recall = rpt['1']['recall']

            df0 = df_[df_['label'] == 0]
            df1 = df_[df_['label'] == 1]

            plt.figure(figsize=(10, 6))

            plt.plot(df0['Rate'], 'o', label=class_names[0])
            plt.plot(df1['Rate'], 'o', label=class_names[1])

            plt.legend()
            plt.ylim([-0.1, 1.1])
            # plt.title(dataset)
            plt.grid(axis='y', linestyle='--')
            plt.yticks([0.0, 0.1, 0.4, 0.5, 0.6, 0.9, 1.0])

            # 显示Acc & TPR
            plt.text(20, 0.5, 'Acc=%.4f' % (acc,))
            plt.text(20, 0.5 - 0.06, 'TPR=%.4f' % (TPR,))
            plt.text(20, 0.5 - 0.06 * 2, 'recall=%.4f' % (recall,))
            plt.text(20, 0.5 - 0.06 * 3, 'Total=%d' % (len(df_),))

            if dataset != 'Black_List':
                # 显示错误的CTL
                for idx_ in df0[df0['Rate'] >= 0.5].index:
                    plt.text(idx_ + 1, df0.loc[idx_, 'Rate'], '%d' % (idx_,))

                # 显示错误的~CTL
                for idx_ in df1[df1['Rate'] <= 0.5].index:
                    plt.text(idx_ + 1, df1.loc[idx_, 'Rate'], '%d' % (idx_,))

            # Black List
            if dataset == 'Black_List':
                # 正确的CTL
                for idx_ in df0[df0['Rate'] < 0.5].index:
                    plt.text(idx_ + 1, df0.loc[idx_, 'Rate'], '%d' % (idx_,))

                # 正确的~CTL
                for idx_ in df1[df1['Rate'] > 0.5].index:
                    plt.text(idx_ + 1, df1.loc[idx_, 'Rate'], '%d' % (idx_,))

            # save fig
            fig_file_path = os.path.join('./Fig/', self.bone_file_name,
                                         self.bone_file_name + '_%d_' % (group_sn,) + dataset + '.png')
            plt.savefig(fig_file_path)

            # confusion matrix & save
            # from pyriemann.utils.viz import plot_confusion_matrix
            from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


            # plt.figure()
            # ConfusionMatrixDisplay.from_estimator(y_true, y_pred, self.class_names, title=dataset)

            cm = confusion_matrix(y_true, y_pred, normalize='true')
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.class_names)
            disp.plot()


            # save fig
            fig_file_path = os.path.join('./Fig/', self.bone_file_name,
                                         self.bone_file_name + '_%d_' % (group_sn,) + dataset + '_confusion.png')
            plt.savefig(fig_file_path)

        # 计算白名单
        # df_wl = df_info # [~df_info['Black_List']] # Flip BL
        df_wl = df_info[(~df_info['Black_List']) | (df_info['label'] == 1)]
        print(df_wl.shape)
        df_plot_(df_wl, 'White_List')

        # # 计算黑名单
        # df_bl = df_info[df_info['Black_List']]
        # df_plot_(df_bl, 'Black_List')

    def evaluate_total_rate_of_fif(self, groups):
        """
        分析'最新'的 self.epochs_file, 评估fif
        :param groups:
        :return:
        """
        from sklearn.metrics import classification_report, confusion_matrix

        evaluate_file = self.bone_file_name + '_evaluate.csv'

        df_info = pd.read_csv(self.info_file, index_col=0)
        # df_wl = df_info[~df_info['Black_List']]

        evaluate_dict = {'TN': [],
                         'FP': [],
                         'TP': [],
                         'FN': [],
                         'Acc': [],
                         'TPR': [],
                         'Recall': []
                         }

        for group in groups:
            epochs_file = self.epochs_file[:-4] + '_' + str(group) + '.csv'
            print(epochs_file)
            df_epochs = pd.read_csv(epochs_file, index_col=0)

            for index in df_info.index:
                df_epochs_of_fif = df_epochs[df_epochs['fif_idx'] == index].copy()
                df_info.loc[index, 'Rate'] = df_epochs_of_fif['Pred'].mean()

            df_info['Pred'] = df_info['Rate'] >= 0.5

            # df_wl = df_info #[~df_info['Black_List']]   # Flip BL
            df_wl = df_info[(~df_info['Black_List']) | (df_info['label'] == 1)]
            print(df_wl.shape)
            y_pred = df_wl['Pred'].to_numpy()
            y_true = df_wl['label'].to_numpy()

            rpt = classification_report(y_true, y_pred, output_dict=True)
            print(rpt)

            evaluate_dict['Acc'].append(rpt['accuracy'])
            evaluate_dict['TPR'].append(rpt['1']['precision'])
            evaluate_dict['Recall'].append(rpt['1']['recall'])

            cm = confusion_matrix(y_true, y_pred)
            print(cm)

            evaluate_dict['TN'].append(cm[0, 0])
            evaluate_dict['TP'].append(cm[1, 1])
            evaluate_dict['FP'].append(cm[0, 1])
            evaluate_dict['FN'].append(cm[1, 0])

        for key in evaluate_dict:
            evaluate_dict[key].append(np.mean(evaluate_dict[key]))
            evaluate_dict[key].append(np.std(evaluate_dict[key][:-1]))

        df_evaluate = pd.DataFrame(evaluate_dict, index=list(groups) + ['mean', 'std'])
        df_evaluate.to_csv(evaluate_file, index=True)

    ########################################################

    def load_feature_data_to_array(self, normalize=False):
        """
        不分割fif成epochs, 将raw_data整体存入字典.
        :return:
        """
        print('>>', 'load_feature_data_to_array()')

        df_dataset_info = pd.read_csv(self.info_file)
        self.dataset_dict = {}

        x_list = []
        l_list = []

        # 逐行处理df_dataset_info
        for idx in df_dataset_info.index:
            row = df_dataset_info.iloc[idx]  # 读一行
            file_path = row['file'].strip()
            label = row['label']

            base_file = os.path.split(file_path)[-1]
            features_file_path = os.path.join(features_path, base_file + '.pkl')
            print(features_file_path)

            with open(features_file_path, 'rb') as f:
                x_all = pickle.load(f)

                if eeg_config['data_transform']:
                    # x_all = transformer_like(x_all)
                    T = eeg_config['T']
                    x_all = SelfAttention(T=T)(x_all)

            if eeg_config['data_normalize']:
                x_all = x_all / np.linalg.norm(x_all, axis=1, keepdims=True)

            x_list.append(x_all)
            l_list += [label] * x_all.shape[0]

        x_array = np.concatenate(x_list, axis=0)
        labels = np.array(l_list)

        self.dataset_array = torch.from_numpy(x_array)
        self.dataset_labels = torch.from_numpy(labels)

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

        return self.alpha_plan[epoch]

    def build_train_loader(self, batch_size=64):
        """
        :return:
        """

        df_epochs = pd.read_csv(self.epochs_file)
        # self.load_feature_data_to_mem(normalize=eeg_config['data_normalize'])

        dataset = eeg_config['dataset']

        train_dataset = dataset(self.df_train,              # df_epochs
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

    def build_loader(self, df_epochs, batch_size=1024):
        """
        :return:
        """

        dataset = eeg_config['dataset']

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

    def train_pytorch_adam(self):
        """
        LPSO
        :return:
        """
        import time

        epochs = eeg_config['epochs']
        batch_size = eeg_config['batch_size']
        epochs_phase_1 = eeg_config['epochs_phase_1']

        # select model
        net = eeg_config['net']
        model = net(self.n_classes, eeg_config['feature_dim'])
        print(model)

        model.cuda(device)

        # define optimizer
        optimizer = torch.optim.Adam(model.parameters(),
                                     weight_decay=eeg_config['weight_decay'],
                                     lr=eeg_config['lr_init'])

        self.define_adam_learning_rate()

        # 自动生成模型文件名
        model_file = self.log.open(model.__class__.__name__)

        # 样本均衡
        class_weight_training_set = self.class_weight_df_PT(self.df_train)
        class_weight_training_set = class_weight_training_set.cuda(device, non_blocking=True)

        # define data loader
        train_loader = self.build_train_loader(batch_size=batch_size)
        val_loader = self.build_loader(self.df_val, batch_size=batch_size)

        # Early Stopping init
        patience = epochs // 2
        early_stopping_cnt = 0

        best_val_acc = 0.0
        for epoch in range(epochs):
            t0 = time.time()

            print("\nEpoch {0}/{1}".format(str(epoch + 1), str(epochs)))

            # 0.1 set learning rate for every epoch
            lr = self.adjust_adam_learning_rate(optimizer, epoch)

            # 0.2 Meter class instantiation
            losses_meter = AverageMeter('Loss', ':.6f')
            acc_meter = AccuracyMeter('Acc', ':.4f')

            # 0.3 !!!
            model.train()

            # 1. 逐个batch进行训练
            for batch_idx, (xs, labels) in enumerate(train_loader):
                xs = xs.cuda(device, non_blocking=True)
                labels = labels.cuda(device, non_blocking=True)

                # Forward + Backward + Optimize
                logits = model(xs)
                loss = F.cross_entropy(logits, labels, class_weight_training_set)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                acc_meter.update(logits.detach().cpu().numpy(), labels.cpu().numpy())
                losses_meter.update(loss.item(), labels.shape[0])

            print(losses_meter, acc_meter)
            print(time.time() - t0)

            # 2. 评估验证集
            val_loss, val_acc = self.evaluate_pytorch(model, val_loader, F.cross_entropy)

            # 3. 记录
            self.log.monitor.update(accuracy=acc_meter.get_value(),
                                    loss=losses_meter.get_value(),
                                    val_accuracy=val_acc,
                                    val_loss=val_loss,
                                    lr=lr
                                    )

            self.log.monitor.report()

            # 4. save thr best model
            if epoch > epochs_phase_1 and val_acc >= best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), model_file[:-3] + '.pt')
                early_stopping_cnt = 0

            # Early Stopping
            early_stopping_cnt += 1
            if early_stopping_cnt > patience:
                break

        # -1. save
        self.log.close()

    def evaluate_pytorch(self, net, data_loader, cost=None):
        losses = AverageMeter('Loss', ':.6f')
        acc = AccuracyMeter('Acc', ':.4f')

        net.eval()
        with torch.no_grad():
            for x, y in data_loader:
                x, y = x.cuda(device, non_blocking=True), y.cuda(device, non_blocking=True)

                y_hat = net(x)
                loss = cost(y_hat, y.long())

                losses.update(loss.item(), x[0].size(0))
                acc.update(y_hat.cpu().numpy(), y.cpu().numpy())
        # print(acc1, acc2)

        return losses.get_value(), acc.get_value()

    def predict_pytorch(self, net, data_loader):
        pred_list = []
        net.eval()
        with torch.no_grad():
            for x, y in data_loader:
                x, y = x.cuda(device, non_blocking=True), y.cuda(device, non_blocking=True)

                y_hat = net(x)
                pred_list.append(y_hat.cpu().numpy())

        preds = np.concatenate(pred_list, axis=0)
        return preds

    def probs_pytorch(self, net, data_loader):
        pred_list = []
        net.eval()
        with torch.no_grad():
            for x, y in data_loader:
                x, y = x.cuda(device, non_blocking=True), y.cuda(device, non_blocking=True)

                y_hat = net(x)
                pred_list.append(y_hat.cpu())

        preds = torch.concat(pred_list, dim=0)
        probs = torch.softmax(preds, dim=1)
        return probs.numpy()


# 添加配置文件
eeg_config = {'duration': 1.5,
              'stride': 1.0,
              'CF': 1,  # !!!!!!

              'samples_balance': False,

              'data_transform': True,
              'data_normalize': False,
              'T': 1.0,  # transformer-like(dk^0.5)

              'epochs': 200,
              'epochs_phase_1': 10,  # 20
              'warmup': 5,
              'epoch_decay_start': 80,

              # 'batch_size': 256 * 4,  # 1024
              # 'lr_init': 5.0e-2 * 4,  # 1.0E-3
              'batch_size': 1024,
              'lr_init': 1.0E-3,

              'net': EEGNet_PT452_FC1,
              'l2_weight': 1.0e-4,
              'weight_decay': 1.0e-4,

              'feature_dim': 512,

              'dataset': DatasetCoTeachingArray,

              'drop_bad_model': True,
              'val_acc_th': 0.85,
              'val_loss_th': 2.0

              }

keyword = 'EC'

# bl_n = [1, 4, 5, 7, 9, 13, 16, 18, 27, 28, 30, 33, 35, 39, 41, 43, 47, 49, 50, 54, 55]
# bl_p = [73, 75, 83, 85, 87, 96, 97, 101, 106, 114, 118, 125, 127]
#
# black_list = bl_n + bl_p + [12, 48] + [91, 122]

black_list_file = './EEGLab_CoT+_FC1_BLA__D512_TC16V3_IMB2__C502_P2_BL.json'
features_path = '../../SimSiam(Imb2)/Features/EEGLab_SimSiam_F512_CP_V_IMB2_TC16V3_1.5_1.0/'


if __name__ == '__main__':
    eeg = EEGLabLOSOFeaturePytorch()
    eeg.register_logging()


    def init():
        eeg.collect_dataset_info_with_black_list()
        # eeg.collect_dataset_info()
        # # eeg.make_epochs_to_disk()

        eeg.make_epochs_sn_df()


    init()

    # 加载 feature 文件数据至 dict
    # eeg.load_feature_data_to_mem()
    eeg.load_feature_data_to_array()

    eeg.fit_method = eeg.train_pytorch_adam

    # k-fold 目标测试集
    for g in range(10):
        eeg.group = g

        # 1. 前期记录 'probability'
        eeg.evaluate_lpso_with_k_fold_feature(n_splits_fold=5,
                                              random_state=2023700 + g * 1000,
                                              n_splits_test_set=5,
                                              random_state_test=2012345 + g * 1000,
                                              P=1)

        # 2. 后期处理, 计算 fif
        eeg.evaluate_rate_of_fif(g)

        # -1. save as
        eeg.save_epochs_info_as(g)

        eeg.evaluate_total_rate_of_fif(range(g+1))

    # # eeg.summary()
    # # eeg.select_black_list()
    #
    # os.system("shutdown -s -t 240")


