
import numpy as np
import pickle

from tensorflow.keras import utils as np_utils
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K
import tensorflow as tf

import pandas as pd
import os
import time


from tensorflow.keras.utils import Sequence


class DataSequence(Sequence):
    """
    基础的数据流生成器，每次迭代返回一个batch,
    可直接用于fit_generator的generator参数.
    fit_generator会将其再次封装为一个多进程的数据流生成器,
    而且能保证在多进程下的一个epoch中不会重复取相同的样本.
    https://www.jb51.net/article/188905.htm
    https://blog.csdn.net/rookie_wei/article/details/100013787
    """
    def __init__(self, x, y=None, batch_size=None, shuffle=True, sample_weight=None, dataset_dict=None, label_noise=0.0, axis=-1, method='sn', one_hot=False):
        """
        自适应输入数据的类型, 包括np.array和Dataframe.
        :param x: np.array or Dataframe
        :param y: 如果x是Dataframe类型, 可忽略
        :param batch_size:
        :param shuffle: 使用model的predict时, 应设置为False
        :param sample_weight: 样本权重 None|np.array
        :param label_noise: DisturbLabel rate [0.0, 0.2]
        :param axis: 合并多个epochs指定的轴, 用于扩展example的高度
        :param method: 合并多个epochs之方法
        :param one_hot: bool, 是否需要one_hot编码
        """
        import math

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sample_weight = sample_weight
        self.label_noise = label_noise
        self.axis = axis
        self.method = method
        self.dataset_dict = dataset_dict
        self.one_hot = one_hot

        if isinstance(x, pd.DataFrame):
            self.df_xy = x
            self.loader = self.load_data_with_df
            self.total = len(self.df_xy)
            self.n_batches = math.ceil(self.total / self.batch_size)        # math.ceil向上取整

            # df包含多个epochs文件的后缀序号
            self.df_columns = x.columns.to_numpy()
            if 'sn_str' in self.df_columns:
                self.loader = self.load_data_with_df_based_on_sn_str
        else:   # np.ndarray
            self.x = x
            self.y = y
            self.loader = self.load_data
            self.n_batches = math.ceil(self.x.shape[0] / self.batch_size)   # math.ceil向上取整
            self.total = self.x.shape[0]

        if self.dataset_dict is not None:
            self.loader = self.load_data_with_df_dict

        self.permutation_pos = np.arange(self.total)
        self.on_epoch_end()

        self.example_method_func = {'random': self._example_by_random_sel,
                                    'shuffle': self._example_by_shuffle,
                                    'sn': self._example_by_sn}
        try:
            self.example_method = self.example_method_func[method]
        except KeyError:
            self.example_method = self.example_method_func['sn']

        # if self.method == 'random':
        #     self.self.example_method = self._example_by_random_sel
        # elif self.method == 'shuffle':
        #     self.self.example_method = self._example_by_shuffle
        # else:
        #     self.self.example_method = self._example_by_sn

        self.table_one_hot = np.array([[1.0, 0.0], [0.0, 1.0]], dtype='float32')

    def __len__(self):
        # 使用math.ceil向上取整
        # 调用len(instance)时返回，是每个epoch我们需要读取数据的次数(mini-batch的数量)
        return self.n_batches

    def _disturb_label(self, y):
        """
        人为加扰label(DisturbLabel)
        :param y:
        :return:
        """
        # noise_indices = np.random.randint(len(indices_), size=max(1, int(len(indices_)*self.label_noise)))    # 至少翻转一个label
        noise_indices = np.random.choice(a=range(y.shape[0]), size=int(y.shape[0]*self.label_noise), replace=False)
        # print(noise_indices)

        # for idx in noise_indices:
        #     y[idx] = 1 - y[idx]
        y[noise_indices] = np.random.randint(2, size=len(noise_indices))    # np.random.binomial(1, 0.5, size=len(noise_indices))

        #     # noise_idx = np.random.randint(len(y))
        #     # y[noise_idx] = 1 - y[noise_idx]
        return y

    def load_data_with_df(self, indices_):
        """
        DataFrame数据的加载.
        :param indices_:
        :return:
        """
        x = []
        y = []
        for idx in indices_:
            file_path = self.df_xy['file'].iloc[idx]
            with open(file_path, 'rb') as f:
                xn, yn = pickle.load(f)
            #xy_npz = np.load(file)
            #xn = xy_npz['x']
            #yn = xy_npz['y']
            #print(xn)
            yn = self.df_xy['label'].iloc[idx]
            x.append(xn)
            y.append(yn)

        x = np.concatenate(x, axis=0)
        y = np.array(y)

        # 人为加扰label(DisturbLabel)
        if self.label_noise > 0.0:
            y = self._disturb_label(y)

        if self.one_hot:
            y = self.table_one_hot[y]

        # 样本权重, 用于evaluate时的样本均衡. 作用可忽略
        if self.sample_weight:
            w = []
            for yi in y:
                w.append(yi)
            return x, y, np.array(w)

        else:
            return x, y

    def _example_by_random_sel(self, idx):
        """
        在当前fif分割出的epochs中, 随机挑选指定数量的epochs, 按指定轴合并成example.
        :param idx:
        :return:
        """
        if self.dataset_dict is not None:
            return self._example_by_random_sel_dict(idx)

        file_path = self.df_xy['file'].iloc[idx]
        sn_str_of_example = self.df_xy['sn_str'].iloc[idx]              # 多个epochs文件的后缀序号, 此函数只使用其数量
        upper_limit_of_example = self.df_xy['upper_limit'].iloc[idx]
        n_epoch = len(sn_str_of_example.split(','))

        # 随机抽取epochs的序号, 加载后按指定轴合并成example
        xn_list = []
        epochs_sn_sel = np.random.choice(a=range(upper_limit_of_example), size=n_epoch, replace=True)  # replace=False @ 221202 样例组合恐过少
        for sn in epochs_sn_sel:
            with open(file_path + str(sn).zfill(4) + '.pkl', 'rb') as f:
                xn_epoch, _ = pickle.load(f)
                xn_list.append(xn_epoch)
        example = np.concatenate(xn_list, axis=self.axis)
        return example

    def _example_by_shuffle(self, idx):
        if self.dataset_dict is not None:
            return self._example_by_shuffle_dict(idx)

        file_path = self.df_xy['file'].iloc[idx]
        sn_str_of_example = self.df_xy['sn_str'].iloc[idx]     # 多个epochs文件的后缀序号

        # 依次加载epochs, 置乱后合并
        xn_list = []
        for sn in sn_str_of_example.split(','):
            with open(file_path + sn + '.pkl', 'rb') as f:
                xn_epoch, _ = pickle.load(f)
                xn_list.append(xn_epoch)

        np.random.shuffle(xn_list)
        example = np.concatenate(xn_list, axis=self.axis)
        return example

    def _example_by_sn(self, idx):
        if self.dataset_dict is not None:
            return self._example_by_sn_dict(idx)

        file_path = self.df_xy['file'].iloc[idx]
        sn_str_of_example = self.df_xy['sn_str'].iloc[idx]     # 多个epochs文件的后缀序号

        # 依次加载后合并
        xn_list = []
        for sn in sn_str_of_example.split(','):
            with open(file_path + sn + '.pkl', 'rb') as f:
                xn_epoch, _ = pickle.load(f)
                xn_list.append(xn_epoch)

        if len(xn_list) > 1:
            example = np.concatenate(xn_list, axis=self.axis)
        else:
            example = xn_list[0]

        return example

    def _example_by_random_sel_dict(self, idx):
        """
        在当前fif分割出的epochs中, 随机挑选指定数量的epochs, 按指定轴合并成example.
        :param idx:
        :return:
        """
        # print('_example_by_random_sel_dict(self, idx)')

        file_path = self.df_xy['file'].iloc[idx]
        sn_str_of_example = self.df_xy['sn_str'].iloc[idx]              # 多个epochs文件的后缀序号, 此函数只使用其数量
        upper_limit_of_example = self.df_xy['upper_limit'].iloc[idx]
        n_epoch = len(sn_str_of_example.split(','))

        x_all = self.dataset_dict[file_path]
        # upper_limit_of_example = x_all.shape[0]

        # 随机抽取epochs的序号, 加载后按指定轴合并成example
        xn_list = []
        # epochs_sn_sel = np.random.choice(a=range(upper_limit_of_example), size=n_epoch, replace=False)  # replace=False @ 221202 样例组合恐过少
        epochs_sn_sel = np.random.choice(a=range(upper_limit_of_example), size=n_epoch, replace=True)  # replace=False @ 221202 样例组合恐过少

        for sn in epochs_sn_sel:
            xn_epoch = x_all[int(sn): int(sn)+1]
            xn_list.append(xn_epoch)
            # with open(file_path + str(sn).zfill(4) + '.pkl', 'rb') as f:
            #     xn_epoch, _ = pickle.load(f)
            #     xn_list.append(xn_epoch)

        example = np.concatenate(xn_list, axis=self.axis)
        return example

    def _example_by_shuffle_dict(self, idx):
        file_path = self.df_xy['file'].iloc[idx]
        sn_str_of_example = self.df_xy['sn_str'].iloc[idx]     # 多个epochs文件的后缀序号

        x_all = self.dataset_dict[file_path]

        # 依次加载epochs, 置乱后合并
        xn_list = []
        for sn in sn_str_of_example.split(','):
            xn_epoch = x_all[int(sn): int(sn)+1]
            xn_list.append(xn_epoch)

        np.random.shuffle(xn_list)

        if len(xn_list) > 1:
            example = np.concatenate(xn_list, axis=self.axis)
        else:
            example = xn_list[0]

        return example

    def _example_by_sn_dict(self, idx):
        file_path = self.df_xy['file'].iloc[idx]
        file_fif = file_path    # os.path.split(file_path)[-1]
        sn_str_of_example = self.df_xy['sn_str'].iloc[idx]     # 多个epochs文件的后缀序号

        x_all = self.dataset_dict[file_fif]

        # 依次加载后合并
        xn_list = []
        for sn in sn_str_of_example.split(','):
            xn_epoch = x_all[int(sn): int(sn)+1]
            xn_list.append(xn_epoch)

        if len(xn_list) > 1:
            example = np.concatenate(xn_list, axis=self.axis)
        else:
            example = xn_list[0]

        return example

    def load_data_with_df_based_on_sn_str(self, indices_):
        """
        DataFrame索引数据的加载.
        df包含多个epochs文件的序号后缀, 需加载后按指定轴合并.
        :param indices_:
        :return:
        """
        x_batch = []
        y_batch = []
        for idx in indices_:
            # file_path = self.df_xy['file'].iloc[idx]
            # sn_str_of_example = self.df_xy['sn_str'].iloc[idx]     # 多个epochs文件的后缀序号
            # upper_limit_of_example = self.df_xy['upper_limit'].iloc[idx]

            # if self.label_noise > 0.0:
            #     # xn = self._example_by_random_sel(idx)
            #     xn = self._example_by_shuffle(idx)
            # else:
            #     xn = self._example_by_sn(idx)
            #
            # # xn = self._example_by_sn(idx)

            xn = self.example_method(idx)
            x_batch.append(xn)

            yn = self.df_xy['label'].iloc[idx]
            y_batch.append(yn)

        x_batch = np.concatenate(x_batch, axis=0)
        y_batch = np.array(y_batch)

        # 人为加扰label(DisturbLabel)
        if self.label_noise > 0.0:
            y_batch = self._disturb_label(y_batch)

        if self.one_hot:
            y_batch = self.table_one_hot[y_batch]

        # 样本权重, 用于evaluate时的样本均衡. 作用可忽略
        if self.sample_weight:
            w = []
            for yi in y_batch:
                w.append(yi)
            return x_batch, y_batch, np.array(w)
        else:
            return x_batch, y_batch

    def load_data_with_df_dict(self, indices_):
        """
        DataFrame索引数据的加载.
        df包含多个epochs文件的序号后缀, 需加载后按指定轴合并.
        :param indices_:
        :return:
        """
        x_batch = []
        y_batch = []
        for idx in indices_:
            # file_path = self.df_xy['file'].iloc[idx]
            # sn_str_of_example = self.df_xy['sn_str'].iloc[idx]     # 多个epochs文件的后缀序号
            # upper_limit_of_example = self.df_xy['upper_limit'].iloc[idx]

            # if self.label_noise > 0.0:
            #     # xn = self._example_by_random_sel(idx)
            #     xn = self._example_by_shuffle(idx)
            # else:
            #     xn = self._example_by_sn(idx)
            #
            # # xn = self._example_by_sn(idx)

            xn = self.example_method(idx)
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

    def load_data(self, indices_):
        """
        np.array数据的加载.
        :param indices_:
        :return:
        """
        x_batch = self.x[indices_]
        y_batch = self.y[indices_]
        return x_batch, y_batch

    def __getitem__(self, idx):
        # instance[idx]
        # math.ceil表示向上取整, 需防止索引越界!
        indices = self.permutation_pos[idx*self.batch_size:min(self.total, (idx+1)*self.batch_size)]
        batch = self.loader(indices)

        return batch

    def on_epoch_end(self):
        # 重写的父类Sequence中的on_epoch_end方法，在每次迭代完后调用。
        # 每次迭代后重新打乱训练集数据
        # print('Testing Shuffle!')
        if self.shuffle:
            np.random.seed(int(time.time()))
            np.random.shuffle(self.permutation_pos)


class DataSequenceDF(Sequence):
    """
    基础的数据流生成器，每次迭代返回一个batch,
    可直接用于fit_generator的generator参数.
    fit_generator会将其再次封装为一个多进程的数据流生成器,
    而且能保证在多进程下的一个epoch中不会重复取相同的样本.
    https://www.jb51.net/article/188905.htm
    https://blog.csdn.net/rookie_wei/article/details/100013787
    """
    def __init__(self, x, batch_size=None, shuffle=True, sample_weight=None, label_noise=0.0, axis=-1):
        """
        输入数据的类型Dataframe.
        :param x: Dataframe
        :param batch_size:
        :param shuffle: 使用model.predict时, 应设置为False
        :param sample_weight: 样本均衡权重字典 None|{}
        :param label_noise: DisturbLabel rate [0.0, 0.2]
        :param axis: 合并多个epochs指定的轴, 用于扩展example的高度
        """
        import math

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sample_weight = sample_weight
        self.label_noise = label_noise
        self.axis = axis

        self.df_xy = x
        self.loader = self.load_data_with_df
        self.total = len(self.df_xy)
        self.n_batches = math.ceil(self.total / self.batch_size)        # math.ceil向上取整

        self.permutation_pos = np.arange(self.total)

        np.random.seed(int(time.time()))
        self.on_epoch_end()

    def __len__(self):
        # 使用math.ceil向上取整
        # 调用len(instance)时返回，是每个epoch我们需要读取数据的次数(mini-batch的数量)
        return self.n_batches

    def load_data_with_df(self, indices_):
        """
        DataFrame数据的加载.
        :param indices_:
        :return:
        """
        x = []
        y = []
        for idx in indices_:
            file_path = self.df_xy['file'].iloc[idx]
            with open(file_path, 'rb') as f:
                xn, yn = pickle.load(f)
            #xy_npz = np.load(file)
            #xn = xy_npz['x']
            #yn = xy_npz['y']
            #print(xn)
            yn = self.df_xy['label'].iloc[idx]
            x.append(xn)
            y.append(yn)

        x = np.concatenate(x, axis=0)
        y = np.array(y)

        # 人为加扰label(DisturbLabel)
        if self.label_noise > 0.0:
            # noise_indices = np.random.randint(len(indices_), size=max(1, int(len(indices_)*self.label_noise)))    # 至少翻转一个label
            noise_indices = np.random.choice(a=range(len(indices_)), size=int(len(indices_)*self.label_noise), replace=False)
            # print(noise_indices)

            # for idx in noise_indices:
            #     y[idx] = 1 - y[idx]
            y[noise_indices] = np.random.randint(2, size=len(noise_indices))    # np.random.binomial(1, 0.5, size=len(noise_indices))

        #     # noise_idx = np.random.randint(len(y))
        #     # y[noise_idx] = 1 - y[noise_idx]

        # 样本权重, 用于evaluate时的样本均衡. 作用可忽略
        if self.sample_weight:
            w = []
            for yi in y:
                w.append(yi)
            return x, y, np.array(w)

        else:
            return x, y

    def __getitem__(self, idx):
        # instance[idx]
        # math.ceil表示向上取整, 需防止索引越界!
        indices = self.permutation_pos[idx*self.batch_size:min(self.total, (idx+1)*self.batch_size)]
        batch = self.loader(indices)

        return batch

    def on_epoch_end(self):
        # 重写的父类Sequence中的on_epoch_end方法，在每次迭代完后调用。
        # 每次迭代后重新打乱训练集数据
        if self.shuffle:
            np.random.shuffle(self.permutation_pos)


class DataSequenceArray(Sequence):
    """
    基础的数据流生成器，每次迭代返回一个batch,
    可直接用于fit_generator的generator参数.
    fit_generator会将其再次封装为一个多进程的数据流生成器,
    而且能保证在多进程下的一个epoch中不会重复取相同的样本.
    https://www.jb51.net/article/188905.htm
    https://blog.csdn.net/rookie_wei/article/details/100013787
    """
    def __init__(self, x, y, batch_size=None, shuffle=True, sample_weight=None, label_noise=0.0):
        """
        np.array类型的数据生成器.
        :param x: np.array
        :param y: label
        :param batch_size:
        :param shuffle: 使用model的predict时, 应设置为False
        :param sample_weight: 样本均衡权重字典 None|{}
        :param label_noise: DisturbLabel rate [0.0, 0.2]
        """
        import math

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sample_weight = sample_weight
        self.label_noise = label_noise

        self.x = x
        self.y = y
        self.n_batches = math.ceil(self.x.shape[0] / self.batch_size)
        self.total = self.x.shape[0]

        self.permutation_pos = np.arange(self.total)
        self.on_epoch_end()

    def __len__(self):
        # 使用math.ceil向上取整
        # 调用len(instance)时返回，是每个epoch我们需要读取数据的次数(mini-batch的数量)
        return self.n_batches

    def load_data(self, indices_):
        """
        np.array数据的加载.
        :param indices_:
        :return:
        """
        x_batch = self.x[indices_]
        y_batch = self.y[indices_]
        return x_batch, y_batch

    def __getitem__(self, idx):
        # 本函数的存在, 可以使用instance[idx]
        # math.ceil表示向上取整, 需防止索引越界!
        indices = self.permutation_pos[idx*self.batch_size:min(self.total, (idx+1)*self.batch_size)]
        batch = self.load_data(indices)

        return batch

    def on_epoch_end(self):
        # 重写的父类Sequence中的on_epoch_end方法，在每次迭代完后调用。
        # 每次迭代后重新打乱训练集数据
        if self.shuffle:
            np.random.seed(int(time.time()))
            np.random.shuffle(self.permutation_pos)


class History(object):
    def __init__(self, epochs, steps, model_file):
        self.epochs = epochs
        self.steps = steps
        self.model_file = model_file

        self.loss = 0.0
        self.acc = 0.0
        self.val_loss = 0.0
        self.val_acc = 0.0
        self.lr = 0.0

        self.current_epoch = 0
        self.et = 0.0

        self.val_acc_best = 0.0

        self.steps_str = str(steps)

        self.history = {'loss': [], 'accuracy': [],
                        'val_loss': [], 'val_accuracy': [],
                        'lr': []}

    def rec_metrics(self, loss, acc, val_loss, val_acc, lr):
        self.loss = loss
        self.acc = acc
        self.val_loss = val_loss
        self.val_acc = val_acc
        self.lr = lr

        self.history['loss'].append(loss)
        self.history['accuracy'].append(acc)
        self.history['val_loss'].append(val_loss)
        self.history['val_accuracy'].append(val_acc)
        self.history['lr'].append(lr)

        if self.val_acc > self.val_acc_best:
            self.best_str = 'val_accuracy improved from {0:.4f} to {1:.4f}, saving model to {2}'.format(self.val_acc_best, val_acc, self.model_file)
            self.val_acc_best = self.val_acc
        else:
            self.best_str = 'val_accuracy not improved {0:.4f}'.format(self.val_acc_best)

    def rec_other(self, current_epoch, elapsed_time):
        self.current_epoch = current_epoch
        self.et = elapsed_time

    def report(self):
        epoch_str = "Epoch {0}:".format(str(self.current_epoch+1).zfill(3))
        step_str = '{}/{}'.format(str(self.steps), str(self.steps))
        et_str = '- {}S'.format(self.et)
        metrics_str = '- loss {0:.4f} - accuracy: {1:.4f} - val_loss: {2:.4f} - val_accuracy: {3:.4f} - lr: {4:.5f}'.format(
            self.loss, self.acc, self.val_loss, self.val_acc, self.lr)

        print(epoch_str, self.best_str)
        print(step_str, et_str, metrics_str)


class Monitor(object):
    def __init__(self, epochs, steps, model_file, val_best_acc=0.0):
        self.epochs = epochs
        self.steps = steps
        self.model_file = model_file

        self.loss = 0.0
        self.acc = 0.0
        self.val_loss = 0.0
        self.val_acc = 0.0
        self.test_loss = 0.0
        self.test_acc = 0.0
        self.lr = 0.0

        self.current_epoch = 0
        self.et = 0.0

        self.val_acc_best = val_best_acc
        self.best_str = None

        self.steps_str = str(steps)

        self.history = {'loss': [], 'accuracy': [],
                        'val_loss': [], 'val_accuracy': [],
                        'test_loss': [], 'test_accuracy': [],
                        'lr': []}

    def rec_metrics(self, loss, acc, val_loss, val_acc, lr, test_loss=0, test_acc=0):
        self.loss = loss
        self.acc = acc
        self.val_loss = val_loss
        self.val_acc = val_acc
        self.lr = lr

        self.history['loss'].append(loss)
        self.history['accuracy'].append(acc)
        self.history['val_loss'].append(val_loss)
        self.history['val_accuracy'].append(val_acc)
        self.history['lr'].append(lr)

        self.history['test_loss'].append(test_loss)
        self.history['test_accuracy'].append(test_acc)

        if self.val_acc > self.val_acc_best:
            self.best_str = 'val_accuracy improved from {0:.4f} to {1:.4f}, saving model to {2}'.format(self.val_acc_best, self.val_acc, self.model_file)
            self.val_acc_best = self.val_acc
        else:
            self.best_str = 'val_accuracy not improved {0:.4f}'.format(self.val_acc_best)

    def rec_other(self, current_epoch, elapsed_time):
        self.current_epoch = current_epoch
        self.et = elapsed_time

    def report(self):
        epoch_str = "Epoch {0}:".format(str(self.current_epoch+1).zfill(3))
        step_str = '{}/{}'.format(str(self.steps), str(self.steps))
        et_str = '- {}S'.format(self.et)
        metrics_str = '- loss {0:.4f} - accuracy: {1:.4f} - val_loss: {2:.4f} - val_accuracy: {3:.4f} - lr: {4:.5f}'.format(
            self.loss, self.acc, self.val_loss, self.val_acc, self.lr)

        print(epoch_str, self.best_str)
        print(step_str, et_str, metrics_str)
        #self.history_plot_save()

    def history_plot_save(self, ylim_acc=None, ylim_loss=None):
        from matplotlib import pyplot as plt

        # 生成fig存盘文件
        fig_file = self.model_file[-3] + 'hist.png'
        fig_file_path = os.path.join('./Fig', fig_file)

        df = pd.DataFrame(self.history)

        plt.figure(figsize=(10, 5), dpi=300)

        plt.subplot(1, 2, 1)
        plt.plot(df[['accuracy', 'val_accuracy']])
        plt.plot(df['val_accuracy'].argmax(), df['val_accuracy'].max(), 'o')
        plt.ylim(ylim_acc)

        plt.subplot(1, 2, 2)
        plt.plot(df[['loss', 'val_loss']])
        plt.plot(df['val_loss'].argmin(), df['val_loss'].min(), 'o')
        plt.ylim(ylim_acc)
        plt.ylim(ylim_loss)

        plt.savefig(fig_file_path)


def mkdir(path):
    import os

    path = path.strip()         # 去除首尾空格
    path = path.rstrip("\\")    # 去除尾部 \ 符号

    if not os.path.exists(path):
        os.makedirs(path)
        return True
    else:
        return False


"""
以下弃用!
"""


class DataSequenceDisk(Sequence):
    """
    基础的数据流生成器，每次迭代返回一个batch,
    可直接用于fit_generator的generator参数.
    fit_generator会将其再次封装为一个多进程的数据流生成器,
    而且能保证在多进程下的一个epoch中不会重复取相同的样本.
    https://www.jb51.net/article/188905.htm
    https://blog.csdn.net/rookie_wei/article/details/100013787
    """
    def __init__(self, df_xy, batch_size, shuffle=True):
        import time
        self.df_xy = df_xy
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.permutation_pos = np.arange(self.df_xy.shape[0])
        self.on_epoch_end()
        self.x = np.zeros(shape=())

    def __len__(self):
        import math
        #math.ceil表示向上取整
        #调用len(BaseSequence)时返回，返回的是每个epoch我们需要读取数据的次数
        #return self.df_xy.shape[0] // self.batch_size
        return math.ceil(self.df_xy.shape[0] / self.batch_size)

    def preprocess_img(self, img_path):
        """
        举例图像样例的预处理, 暂时无用.
        :param img_path:
        :return:
        """
        return
        img = Image.open(img_path)
        resize_scale = self.img_size[0] / max(img.size[:2])
        img = img.resize((self.img_size[0], self.img_size[0]))
        img = img.convert('RGB')
        img = np.array(img)

        # 数据归一化
        img = np.asarray(img, np.float32) / 255.0
        return img

    def load_files(self, indices_):
        x = []
        y = []
        for idx in indices_:
            file = self.df_xy['file'].iloc[idx]
            with open(file, 'rb') as f:
                xn, yn = pickle.load(f)
            #xy_npz = np.load(file)
            #xn = xy_npz['x']
            #yn = xy_npz['y']
            yn = self.df_xy['label'].iloc[idx]
            x.append(xn)
            y.append(yn)
        x = np.concatenate(x, axis=0)
        #y = np_utils.to_categorical(y, num_classes=2).astype('int32')
        y = np.array(y)
        #print(y)

        return x, y

    def __getitem__(self, idx):
        # debug TF2.0
        #if idx == 0 and self.shuffle:
        #    self.on_epoch_end()

        indices = self.permutation_pos[idx*self.batch_size:min(self.df_xy.shape[0], (idx+1)*self.batch_size)]
        x_batch, y_batch = self.load_files(indices)
        #print('epoch', idx)        # 测试多线程
        return x_batch, y_batch

    def on_epoch_end(self):
        """Method called at the end of every epoch.
        """
        # 重写的父类Sequence中的on_epoch_end方法，在每次迭代完后调用。
        # 每次迭代后重新打乱训练集数据
        # np.random.shuffle(self.x_y)
        # self.permutation = list(np.random.permutation(len(self.x)))

        import time

        if self.shuffle:
            np.random.seed(int(time.time()))
            self.permutation_pos = np.random.permutation(self.df_xy.shape[0])
        #print('shuffle')


class DataSequenceDiskX(Sequence):
    """
    基础的数据流生成器，每次迭代返回一个batch,
    可直接用于fit_generator的generator参数.
    fit_generator会将其再次封装为一个多进程的数据流生成器,
    而且能保证在多进程下的一个epoch中不会重复取相同的样本.
    https://www.jb51.net/article/188905.htm
    https://blog.csdn.net/rookie_wei/article/details/100013787
    """
    def __init__(self, df_xy, batch_size):
        self.df_xy = df_xy
        self.batch_size = batch_size

    def __len__(self):
        import math
        return math.ceil(self.df_xy.shape[0] / self.batch_size)

    def load_files(self, indices_):
        x = []
        for idx in indices_:
            file = self.df_xy['file'].iloc[idx]
            with open(file, 'rb') as f:
                xn, _ = pickle.load(f)
            x.append(xn)
        x = np.concatenate(x, axis=0)

        return x

    def __getitem__(self, idx):
        indices = range(idx*self.batch_size, min((idx+1)*self.batch_size, self.df_xy.shape[0]))

        x_batch = self.load_files(indices)
        # print('epoch', idx)        # 测试多线程
        return x_batch


def data_generator(x, y, batch_size):
    """
    自定义的Generator, 用于训练集.
    yield的作用就是把一个函数变成一个generator，
    带有 yield 的函数不再是一个普通函数，Python解释器会将其视为一个generator.
    一个 generator 对象，具有 __next()__ 方法
    https://www.runoob.com/w3cnote/python-yield-used-analysis.html
    """
    total = x.shape[0]
    num_mini_batches = total // batch_size
    while 1:
        permutation = list(np.random.permutation(total))    # 生成置乱索引, 不改变原有数据集
        for m in range(num_mini_batches):
            index = permutation[m*batch_size:(m+1)*batch_size]
            x_batch = x[index]
            y_batch = y[index]
            yield x_batch, y_batch


def data_generator_disk(df_xy, batch_size):
    """
    自定义的Generator, 用于训练集.
    yield的作用就是把一个函数变成一个generator，
    带有 yield 的函数不再是一个普通函数，Python解释器会将其视为一个generator.
    一个 generator 对象，具有 __next()__ 方法
    https://www.runoob.com/w3cnote/python-yield-used-analysis.html
    """
    total = df_xy.shape[0]
    num_mini_batches = total // batch_size

    def load_files(index_):
        x = []
        y = []
        for idx in index_:
            print(idx)
            file = df_xy['file'].iloc[idx]
            print(file)
            with open(file, 'rb') as f:
                xn = pickle.load(f)
            print(xn)
            x.append(xn)
            y.append(df_xy['label'].iloc[idx])
        x = np.concatenate(x, axis=0)
        y = np_utils.to_categorical(y)
        return x, y

    while 1:
        permutation = list(np.random.permutation(total))    # 生成置乱索引, 不改变原有数据集
        for m in range(num_mini_batches):
            index = permutation[m*batch_size:(m+1)*batch_size]
            x_batch, y_batch = load_files(index)
            yield x_batch, y_batch, m
