
import numpy as np
import pandas as pd
import pathlib
import os

import mne
from mne import io
from mne.datasets import sample

from matplotlib import pyplot as plt

from cfg import *


class EEGLabPadding(object):
    def __init__(self):
        self.duration = 2.0
        self.overlap = 0.0

        self.eeg_channels = ['FP1', 'FP2',
                             'F7', 'F3', 'FZ', 'F4', 'F8',
                             'FT7', 'FC3', 'FCZ', 'FC4', 'FT8',
                             'T3', 'C3', 'CZ', 'C4', 'T4',
                             'TP7', 'CP3', 'CPZ', 'CP4', 'TP8',
                             'T5', 'P3', 'PZ', 'P4', 'T6',
                             'O1', 'OZ', 'O2']

        self.eeg_channel_num = len(self.eeg_channels)

        self.pick_channels = None

        self.x_train, self.y_train, self.x_val, self.y_val, self.x_test, self.y_test = (None,)*6

        #  加载文件, 构建数据集
        #self.x_train, self.y_train, self.x_val, self.y_val, self.x_test, self.y_test = self.load_dataset()
        #print(self.x_train.shape, self.x_train.nbytes/1.0e6)

    def check_channels(self):
        """
        遍历所有类别数据, 统计共同的channels
        :return:
        """
        self.pick_channels = None
        channels = None
        for name in sub_name:
            for root, _, files in os.walk(os.path.join(base_path, sub_path[name])):
                for file in files:
                    raw = io.Raw(os.path.join(root, file), preload=False, verbose=False)
                    #df = raw.to_data_frame()
                    #print(df.head())

                    chs = raw.info['ch_names']
                    print(len(chs), chs)
                    if self.pick_channels is None:
                        self.pick_channels = set(chs)
                    else:
                        self.pick_channels = self.pick_channels & set(chs)

                    if channels is None:
                        channels = set(chs)
                    else:
                        channels = channels | set(chs)

                    print('交集', self.pick_channels)
                    print('并集', list(channels))
                    print('差集', channels - set(chs))

        self.pick_channels = list(self.pick_channels)
        print(len(self.pick_channels), self.pick_channels)

    def fix_up_missing_channels(self, raw):
        """
        修补缺失的channels, 旧版本, 结果有误!
        :param raw:
        :return:
        """

        chs = raw.info['ch_names']
        missing_chs = list(set(self.eeg_channels) - set(chs))
        #print('差集', missing_chs)

        if len(missing_chs) == 0:
            return raw

        # pick现有raw中的相等数量channels, 必须使用.copy()完成深拷贝!
        # 主要目的在于, 利用现有raw的info
        raw_slice = raw.copy()
        #raw_slice.pick_channels(chs[0:len(missing_chs)])
        raw_slice.pick(chs[0:len(missing_chs)])

        # rename channels
        mapping = dict(zip(chs[0:len(missing_chs)], missing_chs))
        #print(mapping)
        raw_slice.rename_channels(mapping)

        #data = np.zeros(raw_slice.get_data().shape)
        #raw_slice = mne.io.RawArray(data, raw_slice.info, verbose=False)
        #print(raw_slice.info)

        raw = raw.add_channels([raw_slice])
        raw.info['bads'].extend(missing_chs)
        raw.reorder_channels(self.eeg_channels)
        #print(raw.info)

        raw.interpolate_bads(reset_bads=False, verbose=False, mode='fast', method=dict(meg="MNE", eeg="spline", fnirs="nearest"))
        return raw

    def print_info(self, title='', raw=None):
        print('\n', title)
        print(raw.info['dig'])
        montage = raw.get_montage()
        pos = montage.get_positions()
        print(len(pos['ch_pos']))

    def check_missing_channels_2(self):
        """
        遍历所有类别数据, 统计并尝试补齐缺失channels的方法.
        方法2:
        1. 选择同类别channels完整的raw
        2. 更新待补全raw.info的'dig'
        3. 从通道完整的raw中pick同名的缺失channels
        4. 构建补丁raw骨架
        5. 将待补全raw, 整体add_channels进补丁raw骨架
        6. 在补丁raw中, 将缺失的channels标注为'bad'
        7. 排序补丁raw
        8. interpolation补丁raw
        9. save 补丁raw
        :return:
        """

        for name in sub_name:
            # 搜索同类别(文件夹)中channels完整的raw
            # 实际是假设, 同类别fif文件的'dig'完全相同. 大概率如此!
            dig_list, raw30, montage30 = self.dig_point(name)

            for root, _, files in os.walk(os.path.join(base_path, sub_path[name])):
                for file in files:
                    #fname, ext = os.path.splitext(file)
                    #if ext != '.fif':
                    #    continue

                    # 检查缺失通道状况
                    raw = io.Raw(os.path.join(root, file), preload=True, verbose=False)
                    chs = raw.info['ch_names']
                    missing_chs = list(set(self.eeg_channels) - set(chs))   # 差集

                    if len(missing_chs) > 0:
                        """
                        ValueError: Measurement infos are inconsistent for dig
                        """
                        raw.info['dig'] = raw30.info['dig']

                        raw_slice = raw30.copy()
                        raw_slice.pick_channels(missing_chs)

                        # 构建补丁raw的基础骨架
                        data = np.zeros(shape=(len(missing_chs), raw.n_times))
                        raw_padding = mne.io.RawArray(data, raw_slice.info, verbose=False)
                        self.print_info(raw=raw_padding)

                        # add_channels
                        raw_padding = raw_padding.add_channels([raw])
                        self.print_info(title='after add_channels:', raw=raw_padding)

                        raw_padding.reorder_channels(self.eeg_channels)
                        self.print_info(title='after reorder_channels:', raw=raw_padding)

                        raw_padding.info['bads'].extend(missing_chs)
                        raw_padding.interpolate_bads(reset_bads=True)
                        self.print_info(title='After interpolate_bads:', raw=raw_padding)
                        print(missing_chs)

                        raw_padding.plot(bad_color='r')
                        raw_padding.plot_sensors(show_names=True)
                        raw_padding.plot_psd(average=False)
                        raw_padding.get_montage().plot()

                        plt.show()  # 用于阻塞, 避免程序刷屏！

    def check_missing_channels(self):
        """
        遍历所有类别数据, 统计并尝试补齐缺失channels的方法.
        方法1:
        1. 选择同类别通道完整的raw
        2. 更新待补全raw.info的'dig'
        3. 从通道完整的raw中pick同名的缺失通道
        4. 构建补丁raw
        5. add_channels
        6. 将添加的channels标注为'bad'
        7. 排序
        8. interpolation
        :return:
        """

        for name in sub_name:
            # 搜索同类别(文件夹)中channels完整的raw
            # 实际是假设, 同类别fif文件的'dig'完全相同. 大概率如此!
            dig_list, raw30, montage30 = self.dig_point(name)

            for root, _, files in os.walk(os.path.join(base_path, sub_path[name])):
                for file in files:
                    #fname, ext = os.path.splitext(file)
                    #if ext != '.fif':
                    #    continue

                    # 检查缺失通道状况
                    raw = io.Raw(os.path.join(root, file), preload=True, verbose=False)
                    chs = raw.info['ch_names']
                    missing_chs = list(set(self.eeg_channels) - set(chs))   # 差集

                    if len(missing_chs) > 0:
                        # 修改dig, 补全之. 为后续add_channels做准备
                        # add_channels无法添加新的'dig'子项
                        raw.info['dig'] = dig_list      # raw30.info['dig']
                        #raw.info = raw30.info
                        self.print_info(raw=raw)

                        # 从raw30中pick的名称相同的channels, 必须使用copy()完成深拷贝!
                        # 主要目的在于, 利用raw30的info
                        raw_slice = raw30.copy()
                        raw_slice.pick_channels(missing_chs)

                        # 利用raw的n_times生成空数据, 确保数据shape匹配!
                        # 构建补丁raw
                        data = np.zeros(shape=(len(missing_chs), raw.n_times))
                        raw_padding = mne.io.RawArray(data, raw_slice.info, verbose=False)

                        """ 
                        # 方法2
                        info_slice = raw30.info.copy().pick_channels(missing_chs)
                        data = np.zeros((len(missing_chs), raw.get_data().shape[1]))
                        raw_padding = mne.io.RawArray(data, info_slice, verbose=False)
                        """

                        # add_channels
                        raw = raw.add_channels([raw_padding], force_update_info=True)
                        self.print_info(title='after add_channels:', raw=raw)

                        raw.reorder_channels(self.eeg_channels)
                        self.print_info(title='after reorder_channels:', raw=raw)

                        raw.info['bads'].extend(missing_chs)
                        raw.interpolate_bads(reset_bads=False)
                        self.print_info(title='After interpolate_bads:', raw=raw)
                        print(missing_chs)

                        raw.plot(bad_color='r')
                        #raw.plot_sensors(show_names=True)
                        #raw.plot_psd(average=False)
                        fig = raw.get_montage().plot(kind='3d')
                        fig.gca().view_init(azim=70, elev=15)  # set view angle
                        raw.get_montage().plot(kind='topomap', show_names=False)

                        #fig = mne.viz.plot_alignment(raw.info)
                        #mne.viz.set_3d_view(fig, azimuth=50, elevation=90, distance=0.5)

                        plt.show()  # 用于阻塞, 避免程序刷屏！

    def check_missing_channels_x1(self):
        """
        遍历所有类别数据, 统计并补齐缺失的channels. 用于测试!
        旧版本, 通道插值补全错误!! raw.plot_sensors可以清晰展示
        :return:
        """
        self.pick_channels = None
        channels = None
        for name in sub_name:
            dig_list, raw30, _ = self.dig_point(name)

            for root, _, files in os.walk(os.path.join(base_path, sub_path[name])):
                for file in files:
                    #fname, ext = os.path.splitext(file)
                    #if ext != '.fif':
                    #    continue

                    # 检查缺失通道状况
                    raw = io.Raw(os.path.join(root, file), preload=True, verbose=False)
                    chs = raw.info['ch_names']
                    missing_chs = list(set(self.eeg_channels) - set(chs))
                    #print('差集', missing_chs)

                    if len(missing_chs) > 0:
                        raw.info['dig'] = dig_list

                        # 从现有raw排序考前pick相等数量channels, 必须使用.copy()完成深拷贝!
                        # 主要目的在于, 利用现有raw的info
                        raw_slice = raw.copy()
                        raw_slice.pick_channels(chs[0:len(missing_chs)])
                        #info_slice = info.copy()
                        #raw_slice.pick(chs[0:len(missing_chs)])

                        #print(float(raw.info['sfreq']))
                        #print(raw.get_data().shape)
                        #print(raw.__len__())

                        """
                        # 此种方法看起来, 不灵!
                        #data = np.zeros((len(missing_chs), raw.n_times))
                        data = raw.get_data()[:len(missing_chs), :]
                        info = mne.create_info(ch_names=missing_chs,
                                               ch_types='eeg',
                                               sfreq=raw.info['sfreq'],
                                               verbose=None)

                        bad_channels = mne.io.RawArray(data, info, verbose=False)
                        #print(bad_channels.info)
                        """

                        # rename channels
                        mapping = dict(zip(chs[0:len(missing_chs)], missing_chs))
                        raw_slice.rename_channels(mapping)
                        print('after rename', raw_slice.info['dig'])
                        print(raw_slice.info)

                        # 以下下2行, 对于raw_slice的数据清零. 看似多余!
                        #data = np.zeros(raw_slice.get_data().shape)
                        #raw_slice = mne.io.RawArray(data, raw_slice.info, verbose=False)

                        #raw = raw.add_channels([raw_slice])
                        print('after add', raw.info['dig'])
                        print(raw.info)

                        #raw.info['ch_names'].extend(missing_chs)
                        #raw.info['bads'].extend(missing_chs)
                        raw.reorder_channels(self.eeg_channels)

                        #raw.interpolate_bads(reset_bads=False)
                        #raw.pick_channels([missing_chs[0]])
                        print(missing_chs)

                        raw.plot(bad_color='r')
                        raw.plot_sensors(show_names=True)
                        #raw.plot_psd(average=False)
                        plt.show()  # 用于阻塞, 避免程序刷屏！

    def padding_missing_channels(self):
        """
        遍历所有类别数据, 统计并补齐缺失的channels. 重命名后存盘!
        :return:
        """
        import shutil

        for name in sub_name:
            # 准备通道完整的模板
            dig_list, raw30, _ = self.dig_point(name)

            for root, _, files in os.walk(os.path.join(base_path, sub_path[name])):
                for file in files:
                    fname, ext = os.path.splitext(file)
                    if ext != '.fif':
                        continue

                    #if fname.endswith('padding_raw'):
                        #shutil.move(os.path.join(root, file), os.path.join(base_path, 'bak', file))
                    #    continue

                    # 检查缺失通道状况
                    raw = io.Raw(os.path.join(root, file), preload=True, verbose=False)
                    chs = raw.info['ch_names']
                    missing_chs = list(set(self.eeg_channels) - set(chs))

                    if len(missing_chs) > 0:
                        # 修改dig, 补全之. 为后续add_channels
                        raw.info['dig'] = dig_list
                        print('\nraw:\n', raw.info)
                        print(raw.info['dig'])

                        # pick出raw30中名称相同的channels, 必须使用.copy()完成深拷贝!
                        # 主要目的在于, 利用raw30的info
                        raw_slice = raw30.copy()
                        raw_slice.pick_channels(missing_chs)

                        # 以下下2行, 对于raw_slice的数据清零. 看似多余!
                        data = np.zeros((len(missing_chs), raw.get_data().shape[1]))
                        raw_slice = mne.io.RawArray(data, raw_slice.info, verbose=False)
                        print('\nraw_slice:\n', raw_slice.info)
                        print(raw_slice.info['dig'])

                        # add_channels
                        raw = raw.add_channels([raw_slice], force_update_info=True)
                        print('\nafter add_channels:\n', raw.info)
                        print(raw.info['dig'])

                        raw.info['bads'].extend(missing_chs)

                        raw.reorder_channels(self.eeg_channels)
                        print('\nafter reorder_channels', raw.info)
                        print(raw.info['dig'])

                        raw.interpolate_bads(reset_bads=False)
                        raw.save(fname=pathlib.Path(root, fname[:-3] + 'padding_raw.fif'), overwrite=True)

    def dig_point(self, name, show=False):
        """
        查找相近fif通道完整的raw和DigPoint, 没啥用？!
        已被替代为: find_intact_raw(self, name, show=False)
        :param name:
        :param show:
        :return:
        """

        return self.find_intact_raw(name, show)

        dig_list = {}
        for root, _, files in os.walk(os.path.join(base_path, sub_path[name])):
            for file in files:
                if file.endswith('padding_raw.fif'):
                    continue

                # 显示EEG电极位置
                dig_montage = mne.channels.read_dig_fif(os.path.join(root, file))
                pos = dig_montage.get_positions()#._get_ch_pos()
                print(pos)
                if show:
                    dig_montage.plot()

                raw = io.Raw(os.path.join(root, file), preload=True, verbose=False)

                montage = raw.get_montage()
                pos = montage.get_positions()#._get_ch_pos()
                print(pos)
                #montage.plot()

                chs = raw.info['ch_names']

                if len(chs) == self.eeg_channel_num:
                    print(len(chs), self.eeg_channel_num)
                    dig_montage = mne.channels.read_dig_fif(os.path.join(root, file))
                    #print(dig_montage.get_positions())
                    #dig_montage.get_positions()
                    #dig_montage.plot()
                    print('\nraw30:\n', raw.info)
                    print(raw.info['dig'])
                    dig_list = raw.info['dig']
                    return dig_list, raw, montage
        return None, None

    def find_intact_raw(self, name, show=False):
        """
        查找相近fif文件中, 通道完整的raw和DigPoint.
        :param name:
        :param show: 是否显示输出电极位置
        :return:
        """
        for root, _, files in os.walk(os.path.join(base_path, sub_path[name])):
            for file in files:
                # 过滤已经补全的fif文件
                if file.endswith('padding_raw.fif'):
                    continue

                # 显示EEG电极位置
                dig_montage = mne.channels.read_dig_fif(os.path.join(root, file))
                pos = dig_montage.get_positions()
                print(pos)
                if show:
                    dig_montage.plot(title='mne.channels.read_dig_fif()')

                raw = io.Raw(os.path.join(root, file), preload=True, verbose=False)

                montage = raw.get_montage()
                pos = montage.get_positions()
                print(pos)
                if show:
                    montage.plot(title='raw.get_montage()')

                chs = raw.info['ch_names']

                if len(chs) == self.eeg_channel_num:
                    print('\nraw30:\n', raw.info)
                    print(raw.info['dig'])
                    return raw.info['dig'], raw, montage
        return None, None

    def check_sfreq(self):
        """
        遍历所有类别数据, 统计sfreq
        :return:
        """
        self.pick_channels = None
        for name in sub_name:
            for root, _, files in os.walk(os.path.join(base_path, sub_path[name])):
                for file in files:
                    raw = io.Raw(os.path.join(root, file), preload=False, verbose=False)
                    sfreq = raw.info['sfreq']
                    print(raw.info)

    def check_eeg_dataset(self):
        """
        遍历所有类别数据, raw.info['nchan'], len(raw.info['dig']), raw.info['sfreq']
        :return:
        """
        nchan = []
        nchan_dict = {}
        class_dict = {}
        for name in sub_name:
            for root, _, files in os.walk(os.path.join(base_path, sub_path[name])):
                for file in files:
                    # 过滤修补后的文件
                    if file.endswith('padding_raw.fif'):
                        continue

                    raw = io.Raw(os.path.join(root, file), preload=False, verbose=False)
                    #print(raw.info['nchan'], len(raw.info['dig']), raw.info['sfreq'])
                    nchan.append(raw.info['nchan'])

                    if raw.info['nchan'] in nchan_dict:
                        nchan_dict[raw.info['nchan']] += 1
                    else:
                        nchan_dict[raw.info['nchan']] = 1

                    if name in class_dict:
                        class_dict[name] += 1
                    else:
                        class_dict[name] = 1

        print(class_dict)
        print(nchan_dict)

        # nchan的统计直方图
        plt.hist(nchan, bins=30)
        #plt.plot(class_dict)
        plt.show()


if __name__ == '__main__':
    eeg = EEGLabPadding()

    #eeg.check_sfreq()
    #eeg.check_eeg_dataset()

    #eeg.padding_missing_channels()
    #eeg.check_missing_channels()

    #eeg.check_missing_channels_x1()
    #eeg.check_channels()
    #eeg.check_sfreq()
    #eeg.load_data()

    #eeg.check_eeg_dataset()
