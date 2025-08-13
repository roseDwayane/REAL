
import pandas as pd
import os

import torch


class SimSiamLogging(object):
    """
    维护训练记录, 所有信息均记录在 *_log.csv 文件中.
    """
    def __init__(self, user):
        """
        修改后的初始化函数, 删除了旧版本的参数, 增加了参数user. 使用更加方便.
        使用方法:
            old: log = EEGLabLogging(code_file_name, class_names, self.duration, self.overlap)
            new: log = EEGLabLogging(user=self)
        :param user: self
        """
        self.user = user
        self.bone_file_name = user.bone_file_name
        self.eeg_config = user.eeg_config

        self.log_file = self.bone_file_name + '_log.csv'
        self.pt_csv_path = './pt_csv/'

        mkdir(self.pt_csv_path)

        self.model_file_blank = self.pt_csv_path + self.bone_file_name + '_%s_%d.pt'
        self.history_file_blank = self.pt_csv_path + self.bone_file_name + '_%s_%d.csv'

        self.model_history_name_blank = self.pt_csv_path + self.bone_file_name + '_%s_%d'

        # _log, DataFrame格式
        self.df_log = None

        self.model_file = None
        self.history_file = None
        self.model_history_name = None

    def _append_log_file(self):
        """
        在log文件尾部追加记录(添加一行). 比添加一列要麻烦!
        如果log文件不存在, 则新建一个.
        :return:
        """
        from datetime import datetime

        d = {'date_time': datetime.now().strftime('%Y-%m-%d/%H:%M:%S'),
             'EEG': '/'.join(self.user.class_names),

             'pt_csv': self.model_history_name,
             'epoch': str(0).rjust(5),
             'loss': str(0).ljust(8),
             'argmin': str(0),
             'lr': 0.0}
        d.update(self.eeg_config)

        df_log_new = pd.DataFrame(d, index=[0])

        if os.path.exists(self.log_file):
            self.df_log = pd.read_csv(self.log_file, index_col=0)
            self.df_log = self.df_log.append(df_log_new, ignore_index=True, sort=False)     # 必须用返回值赋值
        else:
            self.df_log = df_log_new

        self.df_log.to_csv(self.log_file)

    def _set_log_file_item(self, key, value, index=-1):
        # print(key, value)
        df_log = pd.read_csv(self.log_file, index_col=0)

        if isinstance(value, float):
            value = ('%.5f' % (value, )).ljust(8)
        elif isinstance(value, int):
            value = str(value)

        df_log.loc[df_log.index[index], key] = value

        df_log.to_csv(self.log_file)

    def _get_new_record_sn_from_log_file(self):
        """
        自动计算h5文件和history文件的序号, 用于添加历史记录.
        :return:
        """
        if os.path.exists(self.log_file):
            self.df_log = pd.read_csv(self.log_file, index_col=0)
            return self.df_log.shape[0]
        else:
            return 0

    def gen_model_history_file_name_from_log_file(self, model_name):
        """
        自动生成h5文件和history文件名, 用于添加历史记录.
        :return:
        """
        sn = self._get_new_record_sn_from_log_file()
        self.model_history_name = self.model_history_name_blank % (model_name, sn)

        self.model_file = self.model_history_name + '.pt'
        self.history_file = self.model_history_name + '.csv'

        self._append_log_file()
        return self.model_file, self.history_file

    def get_transform_base_from_log_file(self, index=-1):
        """
        根据index, 在_log.csv中提取final模型文件名和历史文件名
        :param index: _log.csv中的索引, -1表示最后一行
        :return:
        """
        df_log = pd.read_csv(self.log_file, index_col=0)
        #h5_file = df_log.iloc[index, 'h5_end']
        h5_file = df_log['h5'].iloc[index][:-3] + 'e.pt'

        csv_file = df_log['csv'].iloc[index]
        df_history = pd.read_csv(csv_file)
        return h5_file, df_history

    def get_model_history_from_log_file(self, index=-1):
        """
        根据index, 在_log.csv中提取模型文件名和历史文件名
        :param index: _log.csv中的索引, -1表示最后一行
        :return:
        """
        df_log = pd.read_csv(self.log_file, index_col=0)

        h5_csv_name = df_log['h5_csv'].iloc[index]

        h5_file = h5_csv_name + '.h5'
        h5_final_file = h5_csv_name + 'e.h5'
        csv_file = h5_csv_name + '.csv'

        df_history = pd.read_csv(csv_file)

        return h5_file, df_history, h5_final_file

    def save_final_model_history(self, model_final, df_history, df_history_base=None):
        """
        保存训练结束时的最终model(不一定是最佳), 用于可能的后续迁移学习.
        :param model_final:
        :param df_history:
        :param df_history_base: 旧模型的历史数据
        :return:
        """
        from datetime import datetime

        # model_final_file = self.df_log['pt_csv'].iloc[-1] + 'e.pt'
        # model_final.save(model_final_file)

        if df_history_base is not None:
            df_history = df_history_base.append(df_history, ignore_index=True, sort=False)

        pos = df_history['loss'].argmin()
        self._set_log_file_item('loss', df_history['loss'].iloc[pos])
        self._set_log_file_item('argmin', int(pos))

        # df_history可能没有学习率数据
        try:
            self._set_log_file_item('lr', df_history['lr'].iloc[pos])
        except Exception as e:
            print(e)

        self._set_log_file_item('date_time', datetime.now().strftime('%Y-%m-%d/%H:%M:%S'))
        self._set_log_file_item('epoch', int(df_history.shape[0]))

        df_history.to_csv(self.history_file, index=False)

    def get_last_model_index(self):
        df_log = pd.read_csv(self.log_file, index_col=0)
        return len(df_log) - 1

    def get_latest_model_index(self):
        return self.get_last_model_index()


def mkdir(path):
    import os

    path = path.strip()         # 去除首尾空格
    path = path.rstrip("\\")    # 去除尾部 \ 符号

    if not os.path.exists(path):
        os.makedirs(path)
        return True
    else:
        return False
