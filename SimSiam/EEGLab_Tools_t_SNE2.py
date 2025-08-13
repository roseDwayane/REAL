# mne imports
import mne
from mne import io

# EEGNet-specific imports

# tools for plotting confusion matrices

# PyRiemann imports

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from cfg6 import *
import logging

from SimSiamLib.EEGLab_Tools import *
from SimSiamLib.EEGLab_Transform import *


class EEGLabSimSiamSNE(object):
    """
    SimSiam 提取特征.
    """

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

        # dataFrame数据结构
        self.df_dataset_info = None

    ######################################################

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

                    raw = io.Raw(file_path, preload=False, verbose=False)
                    if len(raw.info['ch_names']) < self.eeg_channel_num:
                        continue

                    # # EC | EO
                    # if not file.startswith(keyword):
                    #     continue
                    if file[:2] not in keyword:
                        continue

                    xn = raw.get_data()

                    xn_std = np.sqrt(np.mean(np.square(xn)))
                    df_dict['std'].append(xn_std)

                    c_std = np.std(xn, axis=1)
                    xn_std = c_std.mean()
                    df_dict['std2'].append(xn_std)

                    xn_std = np.std(xn)
                    df_dict['std3'].append(xn_std)

                    # file_path = file_path.ljust(75)  # 补空格, 目的在于csv文件的可读性.
                    # files_list.append(file_path)
                    print(file_path)

                    df_dict['n_times'].append(raw.n_times)
                    df_dict['EC-EO'].append(file[:2])

                    df_dict['file'].append(file_path)
                    df_dict['eeg_name'].append(eeg_name)
                    df_dict['label'].append(eeg_class_labels[eeg_name])

        df_dataset_info = pd.DataFrame(df_dict, index=[n for n in range(len(df_dict['file']))])

        df_dataset_info.to_csv(self.info_file)

    ######################################################

    def _t_SNE_single(self, feature, labels, name='2D'):
        from sklearn.manifold import TSNE

        features_name = os.path.split(features_path)[-1]
        fig_path = os.path.join('./Fig_t_SNE', features_name, 'single')
        mkdir(fig_path)

        npz_file = os.path.join(fig_path, name + '.npz')

        if not os.path.exists(npz_file):
            tsne = TSNE(n_components=2, verbose=2,
                        learning_rate=200,
                        n_iter=2000, n_iter_without_progress=500)

            feature_tsne = tsne.fit_transform(feature)
            print(feature_tsne.shape)

            # 保存 TSNE 数据
            np.savez(os.path.join(fig_path, name), feature_tsne, labels)
        else:
            data = np.load(npz_file)
            print(data.files)
            feature_tsne, labels = data['arr_0'], data['arr_1']

        plt.figure(name, figsize=(8, 6))

        # # plt.scatter(feature_tsne[:, 0], feature_tsne[:, 1], c=pos)
        # plt.scatter(feature_tsne[pos == 0, 0], feature_tsne[pos == 0, 1], c='r')
        # plt.scatter(feature_tsne[pos == 1, 0], feature_tsne[pos == 1, 1], c='g')
        # plt.scatter(feature_tsne[pos == 2, 0], feature_tsne[pos == 2, 1], c='b')
        print(type(labels), labels)

        # 逐label绘制散点图
        cm = ['k', 'r', 'g', 'y', 'g']
        for n in range(max(labels)+1):
            plt.scatter(feature_tsne[labels == n, 0], feature_tsne[labels == n, 1],
                        c=cm[n],
                        s=5,
                        label=n)

        if indices is not None:
            for n in range(indices.max() + 1):
                feature_tsne_n = feature_tsne[indices == n, :]
                x = feature_tsne_n[:, 0].mean()
                y = feature_tsne_n[:, 1].mean()
                plt.text(x, y, str(n), color='green')

        plt.legend()
        plt.title(features_name + '\\' + name)

        fig_name = name + '.png'
        plt.savefig(os.path.join(fig_path, fig_name), dpi=600)

        # plt.show()

    def _t_SNE(self, feature, labels, name='2D', indices=None):
        from sklearn.manifold import TSNE

        features_name = os.path.split(features_path)[-1]
        fig_path = os.path.join('./Fig_t_SNE', features_name)
        mkdir(fig_path)

        fig_path_single = os.path.join('./Fig_t_SNE', features_name, 'single')
        mkdir(fig_path_single)

        print(type(labels), labels)

        npz_file = os.path.join(fig_path, name + '.npz')

        # tsne = TSNE(n_components=2, learning_rate=50,
        #             # method='exact',
        #             init='random', verbose=2)

        if not os.path.exists(npz_file):
            tsne = TSNE(n_components=2, verbose=2,
                        learning_rate=200,
                        n_iter=2000, n_iter_without_progress=500)

            feature_tsne = tsne.fit_transform(feature)
            print(feature_tsne.shape)

            # 保存 TSNE 数据
            np.savez(os.path.join(fig_path, name), feature_tsne, labels)
        else:
            data = np.load(npz_file)
            print(data.files)
            feature_tsne, labels = data['arr_0'], data['arr_1']

        plt.figure(name, figsize=(2.5, 2), dpi=600)
        plt.subplots_adjust(left=0.10, bottom=0.125, right=0.9, top=0.9, wspace=None, hspace=None)

        # # plt.scatter(feature_tsne[:, 0], feature_tsne[:, 1], c=pos)
        # plt.scatter(feature_tsne[pos == 0, 0], feature_tsne[pos == 0, 1], c='r')
        # plt.scatter(feature_tsne[pos == 1, 0], feature_tsne[pos == 1, 1], c='g')
        # plt.scatter(feature_tsne[pos == 2, 0], feature_tsne[pos == 2, 1], c='b')
        print(type(labels), labels)

        # 逐label绘制散点图
        cm = ['k', 'r', 'g', 'y', 'g']
        for n in range(max(labels)+1):
            plt.scatter(feature_tsne[labels == n, 0], feature_tsne[labels == n, 1],
                        c=cm[n],
                        s=0.1,
                        label=class_names[n])

        if indices is not None:
            for n in range(indices.max() + 1):
                feature_tsne_n = feature_tsne[indices == n, :]
                x = feature_tsne_n[:, 0].mean()
                y = feature_tsne_n[:, 1].mean()
                plt.text(x, y, str(n), color='green')

        # plt.legend()
        plt.legend(shadow=True, prop=text_font)
        # plt.title(features_name + '\\' + name)
        plt.yticks(font=ticks_font['family'], fontsize=ticks_font['size'])
        plt.xticks(font=ticks_font['family'], fontsize=ticks_font['size'])

        fig_name = name + '.png'
        plt.savefig(os.path.join(fig_path, fig_name), dpi=600)

        # plt.show()

    def t_sne_single_fif(self, g=2, SA=False):
        """
        t-SNE降维, 可视化每个 fif 之 epochs 的 feature
        :return:
        """
        from tqdm import tqdm

        df_dataset_info = pd.read_csv(self.info_file)
        df_dataset_info = df_dataset_info[df_dataset_info['EC-EO'] == keyword[0]]

        for index in tqdm(df_dataset_info.index, colour='green'):
            row = df_dataset_info.loc[index]
            file_path = row['file']

            base_file = os.path.split(file_path)[-1]
            features_file_path = os.path.join(features_path, base_file + '.pkl')
            with open(features_file_path, 'rb') as f:
                features = pickle.load(f)

                if SA:
                    features = SelfAttention(T=1.0)(features)

                # 按时间先后分组
                g_list = np.array_split(np.arange(features.shape[0]), g)

                flags_list = []
                for g_array, flag in zip(g_list, range(g)):
                    g_array[:] = flag
                    print(g_array)
                    flags_list.append(g_array)
                flags = np.concatenate(flags_list)

                if SA:
                    self._t_SNE_single(features, flags, name='SA_%s_%s' % (g, index))
                else:
                    self._t_SNE_single(features, flags, name='nSA_%s_%s' % (g, index))

    def t_sne_total_fifs(self, T=1.0, SA=False, indices=False):
        """
        t-SNE降维, 可视化feature
        :param T:
        :param SA:
        :param indices:
        :return:
        """
        from tqdm import tqdm

        df_dataset_info = pd.read_csv(self.info_file)

        # # EC only
        # df_dataset_info = df_dataset_info[df_dataset_info['EC-EO'] == keyword[0]]

        labels = []
        eyes = []
        feature_list = []
        indices_list = []
        for index in tqdm(df_dataset_info.index):
            row = df_dataset_info.loc[index]
            file_path = row['file']
            label = row['label']
            eye = row['EC-EO']

            base_file = os.path.split(file_path)[-1]
            features_file_path = os.path.join(features_path, base_file + '.pkl')
            with open(features_file_path, 'rb') as f:
                features = pickle.load(f)

            # if SA and label == 1:
            # if SA and label == 0:
            if SA:
                features = SelfAttention(T=T)(features)

            feature_list.append(features)
            labels += [label] * features.shape[0]
            indices_list += [index] * features.shape[0]

            eyes += [eye] * features.shape[0]

        feature = np.concatenate(feature_list, axis=0)
        labels = np.array(labels)

        eyes = np.array(eyes)

        if indices:
            indices = np.array(indices_list)
        else:
            indices = None

        # if SA:
        #     self._t_SNE(feature, labels, name='Total_fifs_SA', indices=indices)
        # else:
        #     self._t_SNE(feature, labels, name='Total_fifs_nSA')

        if SA:
            self._t_SNE_all(feature, labels, eyes, name='Total_fifs_SA', indices=indices)
        else:
            self._t_SNE_all(feature, labels, eyes, name='Total_fifs_nSA')

    def _t_SNE_all(self, feature, labels, eyes, name='2D', indices=None):
        from sklearn.manifold import TSNE

        features_name = os.path.split(features_path)[-1]
        fig_path = os.path.join('./Fig_t_SNE', features_name)
        mkdir(fig_path)

        fig_path_single = os.path.join('./Fig_t_SNE', features_name, 'single')
        mkdir(fig_path_single)

        print(type(labels), labels)

        npz_file = os.path.join(fig_path, name + '.npz')

        # tsne = TSNE(n_components=2, learning_rate=50,
        #             # method='exact',
        #             init='random', verbose=2)

        if not os.path.exists(npz_file):
            tsne = TSNE(n_components=2, verbose=2,
                        learning_rate=200,
                        n_iter=2000, n_iter_without_progress=500)

            feature_tsne = tsne.fit_transform(feature)
            print(feature_tsne.shape)

            # 保存 TSNE 数据
            np.savez(os.path.join(fig_path, name), feature_tsne, labels)
        else:
            data = np.load(npz_file)
            print(data.files)
            feature_tsne, labels = data['arr_0'], data['arr_1']

        plt.figure(name, figsize=(2.5, 2), dpi=600)
        plt.subplots_adjust(left=0.10, bottom=0.125, right=0.9, top=0.9, wspace=None, hspace=None)

        print(type(labels), labels)

        feature_tsne_ec = feature_tsne[eyes == 'EC', :]
        labels_ec = labels[eyes == 'EC']

        # 逐label绘制散点图
        cm = ['k', 'r', 'g', 'y', 'g']
        for n in set(labels):
            plt.scatter(feature_tsne_ec[labels_ec == n, 0], feature_tsne_ec[labels_ec == n, 1],
                        c=cm[n],
                        s=0.1,
                        label='EC_' + class_names[n])

        feature_tsne_eo = feature_tsne[eyes == 'EO', :]
        labels_eo = labels[eyes == 'EO']

        # 逐label绘制散点图
        cm = ['gray', 'pink', 'g', 'y', 'g']
        for n in set(labels):
            plt.scatter(feature_tsne_eo[labels_eo == n, 0], feature_tsne_eo[labels_eo == n, 1],
                        c=cm[n],
                        s=0.1,
                        label='EO_' + class_names[n])

        if indices is not None:
            for n in set(indices):
                feature_tsne_n = feature_tsne[indices == n, :]
                x = feature_tsne_n[:, 0].mean()
                y = feature_tsne_n[:, 1].mean()
                plt.text(x, y, str(n), color='green',
                         font=text_font['family'],
                         fontsize=text_font['size'])

        # plt.legend()
        plt.legend(shadow=True, prop=text_font)
        # plt.title(features_name + '\\' + name)
        plt.yticks(font=ticks_font['family'], fontsize=ticks_font['size'])
        plt.xticks(font=ticks_font['family'], fontsize=ticks_font['size'])

        fig_name = name + '.png'
        plt.savefig(os.path.join(fig_path, fig_name), dpi=600)

        # plt.show()


keyword = ['EC', 'EO']
# keyword = ['EC']

# 手工添加
features_path = None    #


def get_dir_name():
    from tkinter import filedialog
    import tkinter as tk

    # 创建打开文件夹窗口
    fold = tk.Tk()
    fold.withdraw()

    dir_name = filedialog.askdirectory(initialdir='./features')
    print(dir_name)

    return dir_name


label_font = {'family': 'Times New Roman', 'size': 7.5}
ticks_font = {'family': 'Times New Roman', 'size': 5}
text_font = {'family': 'Times New Roman', 'size': 3}


if __name__ == '__main__':
    eeg = EEGLabSimSiamSNE()

    features_path = get_dir_name()
    # print(features_path)

    def init():
        eeg.collect_dataset_info()

    # step 1
    init()

    # step 2
    # eeg.t_sne_single_fif()
    # eeg.t_sne_single_fif(SA=True)

    eeg.t_sne_total_fifs(SA=False)
    eeg.t_sne_total_fifs(SA=True, indices=True)



