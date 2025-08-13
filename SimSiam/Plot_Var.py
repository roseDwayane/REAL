
# mne imports
import mne
from mne import io

# EEGNet-specific imports

# tools for plotting confusion matrices

# PyRiemann imports

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

from cfg6 import *
import logging
import os
from cfg6 import *

from EEGLab.EEGLab_Transform import *


def mkdir(path):
    import os

    path = path.strip()         # 去除首尾空格
    path = path.rstrip("\\")    # 去除尾部 \ 符号

    if not os.path.exists(path):
        os.makedirs(path)
        return True
    else:
        return False


label_font = {'family': 'Times New Roman', 'size': 7.5}   # 六号
ticks_font = {'family': 'Times New Roman', 'size': 5}
text_font = {'family': 'Times New Roman', 'size': 2}
legend_font = {'family': 'Times New Roman', 'size': 5}


class PlotVar(object):
    def __init__(self):
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

    def collect_dataset_info(self):
        """
        加载数据集文件, 基础数据目录存盘为 *_info.csv.
        :return: None
        """

        # 用于生成DataFrame(字典生成方法)
        df_dict = {'file': [], 'eeg_name': [], 'EC-EO': [], 'label': [], 'n_times': []}

        # 收集数据集, df_dict
        for eeg_name in eeg_class_names:
            files_list = []
            for root, _, files in os.walk(os.path.join(base_path, eeg_class_paths[eeg_name])):
                for file in files:
                    file_path = os.path.join(root, file)

                    raw = io.Raw(file_path, preload=False, verbose=False)
                    if len(raw.info['ch_names']) < 30:
                        continue

                    # # EC | EO
                    # if not file.startswith(keyword):
                    #     continue
                    if file[:2] not in keyword:
                        continue

                    print(file_path)

                    df_dict['n_times'].append(raw.n_times)
                    df_dict['EC-EO'].append(file[:2])

                    df_dict['file'].append(file_path)
                    df_dict['eeg_name'].append(eeg_name)
                    df_dict['label'].append(eeg_class_labels[eeg_name])

        self.df_dataset_info = pd.DataFrame(df_dict, index=[n for n in range(len(df_dict['file']))])

        return self.df_dataset_info

    def _var(self, x):
        C = np.mean(x, axis=0, keepdims=True)
        diff = x - C
        diff2 = np.square(diff)
        var = np.sum(diff2) / x.shape[0]
        return var, C

    def plot_var(self):
        import pickle

        features_name = os.path.split(features_path)[-1]
        fig_path = os.path.join('./Fig', features_name)
        mkdir(fig_path)

        df_dataset_info = self.df_dataset_info  # pd.read_csv(self.info_file)
        df_dataset_info['var'] = 0

        C_list = []
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

            # C = np.mean(x_all, axis=0, keepdims=True)
            # diff = x_all - C
            # diff2 = np.square(diff)
            # var = np.sum(diff2) / x_all.shape[0]
            # print(C.shape, diff2.shape)
            var, C = self._var(x_all)

            df_dataset_info.loc[idx, 'var'] = var
            C_list.append(C)

        C_all = np.concatenate(C_list, axis=0)
        var2, _ = self._var(C_all)

        plt.figure('Var w/o SA', figsize=(2.5, 1.0))
        plt.subplot(2, 1, 1)
        plt.semilogy(df_dataset_info['var'],
                     label='Var=%.4f/%.2f/%.2f' % (df_dataset_info['var'].mean(), var2, var2/df_dataset_info['var'].mean()))
        # plt.plot(df_dataset_info['label']*10000)
        # plt.ylim([0, min(2.5, df_dataset_info['var'].max()*1.1)])
        plt.legend(loc=1)

        plt.subplot(2, 1, 2)
        plt.hist(df_dataset_info['var'], bins=100)

        fig_name = 'Var_nSA.png'
        plt.savefig(os.path.join(fig_path, fig_name),
                    dpi=300
                    )

    def plot_var_save(self):
        import pickle

        features_name = os.path.split(features_path)[-1]
        fig_path = os.path.join('./Fig', features_name)
        mkdir(fig_path)

        df_dataset_info = self.df_dataset_info  # pd.read_csv(self.info_file)
        df_dataset_info['var'] = 0

        C_list = []
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

            # C = np.mean(x_all, axis=0, keepdims=True)
            # diff = x_all - C
            # diff2 = np.square(diff)
            # var = np.sum(diff2) / x_all.shape[0]
            # print(C.shape, diff2.shape)
            var, C = self._var(x_all)

            df_dataset_info.loc[idx, 'var'] = var
            C_list.append(C)

        C_all = np.concatenate(C_list, axis=0)
        var2, _ = self._var(C_all)

        plt.figure('Var w/o SA save', figsize=(2.5, 1.5))
        plt.subplots_adjust(left=0.15, bottom=0.2, right=0.95, top=0.95)

        plt.semilogy(df_dataset_info['var'],
        # plt.plot(df_dataset_info['var'],
                     linewidth=0.5,
                     color='darkgreen',
                     # label='Var=%.4f/%.2f/%.2f' % (df_dataset_info['var'].mean(), var2, var2/df_dataset_info['var'].mean()))
                     label='var = %.4f/%.2f' % (df_dataset_info['var'].mean(), var2))
        plt.legend(shadow=False, prop=legend_font, frameon=True)

        plt.xticks(font=ticks_font['family'], fontsize=ticks_font['size'])
        plt.yticks(font=ticks_font['family'], fontsize=ticks_font['size'])
        plt.ylabel('var', font=label_font['family'], fontsize=label_font['size'], labelpad=0)
        plt.xlabel('ID', font=label_font['family'], fontsize=label_font['size'], labelpad=0)

        plt.ylim([0.000001, 3.05])

        fig_name = 'Z_Var_nSA.png'
        plt.savefig(fig_name,
                    dpi=600
                    )

        plt.figure('Hist w/o SA save', figsize=(2.5, 1.5))
        plt.subplots_adjust(left=0.15, bottom=0.2, right=0.95, top=0.95)
        plt.hist(df_dataset_info['var'], bins=100)

        plt.xticks(font=ticks_font['family'], fontsize=ticks_font['size'])
        plt.yticks(font=ticks_font['family'], fontsize=ticks_font['size'])
        plt.xlabel('Var', font=label_font['family'], fontsize=label_font['size'], labelpad=0)
        plt.ylabel('Counts', font=label_font['family'], fontsize=label_font['size'], labelpad=0)

        fig_name = 'Z_Hist_nSA.png'
        plt.savefig(fig_name,
                    dpi=600
                    )

    def plot_var_sa(self):
        import pickle

        features_name = os.path.split(features_path)[-1]
        fig_path = os.path.join('./Fig', features_name)
        mkdir(fig_path)

        df_dataset_info = self.df_dataset_info  # pd.read_csv(self.info_file)
        df_dataset_info['var'] = 0

        C_list = []
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
                x_all = SelfAttention(T=1.0)(x_all)

            # C = np.mean(x_all, axis=0, keepdims=True)
            # diff = x_all - C
            # diff2 = np.square(diff)
            # var = np.sum(diff2) / x_all.shape[0]
            # print(C.shape, diff2.shape)
            var, C = self._var(x_all)

            df_dataset_info.loc[idx, 'var'] = var
            C_list.append(C)

        C_all = np.concatenate(C_list, axis=0)
        var2, _ = self._var(C_all)

        plt.figure('Var w/ SA', figsize=(8, 5))
        plt.subplot(2, 1, 1)
        plt.semilogy(df_dataset_info['var'],
                     color='darkgreen',
                     label='Var=%.4f/%.2f/%.2f' % (df_dataset_info['var'].mean(), var2, var2/df_dataset_info['var'].mean()))
        # plt.plot(df_dataset_info['label']*10000)
        # plt.ylim([0, min(2.5, df_dataset_info['var'].max()*1.1)])
        plt.legend(loc=1)

        plt.subplot(2, 1, 2)
        plt.hist(df_dataset_info['var'], bins=100)

        fig_name = 'Var_SA.png'

        plt.savefig(os.path.join(fig_path, fig_name),
                    dpi=300
                    )

    def plot_var_sa_save(self):
        import pickle

        features_name = os.path.split(features_path)[-1]
        fig_path = os.path.join('./Fig', features_name)
        mkdir(fig_path)

        df_dataset_info = self.df_dataset_info  # pd.read_csv(self.info_file)
        df_dataset_info['var'] = 0

        C_list = []
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
                x_all = SelfAttention(T=1.0)(x_all)

            # C = np.mean(x_all, axis=0, keepdims=True)
            # diff = x_all - C
            # diff2 = np.square(diff)
            # var = np.sum(diff2) / x_all.shape[0]
            # print(C.shape, diff2.shape)
            var, C = self._var(x_all)

            df_dataset_info.loc[idx, 'var'] = var
            C_list.append(C)

        C_all = np.concatenate(C_list, axis=0)
        var2, _ = self._var(C_all)

        plt.figure('Var2 w/o SA save', figsize=(2.5, 1.5))
        plt.subplots_adjust(left=0.15, bottom=0.2, right=0.95, top=0.95)

        plt.semilogy(df_dataset_info['var'],
        # plt.plot(df_dataset_info['var'],
                     linewidth=0.5,
                     color='darkgreen',
                     # label='Var=%.4f/%.2f/%.2f' % (df_dataset_info['var'].mean(), var2, var2/df_dataset_info['var'].mean()))
                     label='var = %.4f/%.2f' % (df_dataset_info['var'].mean(), var2))
        plt.legend(shadow=False, prop=legend_font, frameon=True)

        plt.xticks(font=ticks_font['family'], fontsize=ticks_font['size'])
        plt.yticks(font=ticks_font['family'], fontsize=ticks_font['size'])
        plt.ylabel('var', font=label_font['family'], fontsize=label_font['size'], labelpad=0)
        plt.xlabel('ID', font=label_font['family'], fontsize=label_font['size'], labelpad=0)

        plt.ylim([0.000001, 3.05])

        fig_name = 'Z_Var_SA.png'
        plt.savefig(fig_name,
                    dpi=600
                    )

        plt.figure('Hist2 w/o SA save', figsize=(2.5, 1.5))
        plt.subplots_adjust(left=0.15, bottom=0.2, right=0.95, top=0.95)
        plt.hist(df_dataset_info['var'], bins=100)

        plt.xticks(font=ticks_font['family'], fontsize=ticks_font['size'])
        plt.yticks(font=ticks_font['family'], fontsize=ticks_font['size'])
        plt.xlabel('Var', font=label_font['family'], fontsize=label_font['size'], labelpad=0)
        plt.ylabel('Counts', font=label_font['family'], fontsize=label_font['size'], labelpad=0)

        fig_name = 'Z_Hist_SA.png'
        plt.savefig(fig_name,
                    dpi=600
                    )


def get_dir_name():
    from tkinter import filedialog
    import tkinter as tk

    # 创建打开文件夹窗口
    fold = tk.Tk()
    fold.withdraw()

    dir_name = filedialog.askdirectory(initialdir='./features',
                                       title='选择特征数据文件夹'
                                       )
    print(dir_name)

    return dir_name


keyword = ['EC']
features_path = None


if __name__ == '__main__':
    eeg = PlotVar()
    eeg.collect_dataset_info()

    features_path = get_dir_name()

    # eeg.plot_var()
    # eeg.plot_var_sa()

    eeg.plot_var_save()
    eeg.plot_var_sa_save()

    plt.show()
