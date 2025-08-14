
import numpy as np
import pickle

import pandas as pd
import os
import time


def adjust_learning_rate(optimizer, init_lr, epoch, epochs):
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


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
    #     self.reset()
    #
    # def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def get_value(self):
        return self.avg

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class AccuracyMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt

        self.y_hat = None
        self.y = None
        self.acc = 0.0
        self.sum = 0
        self.count = 0

    def update(self, y_hat, y):
        if self.y is None:
            self.y = y
        else:
            self.y = np.concatenate([self.y, y], axis=0)

        if self.y_hat is None:
            self.y_hat = y_hat
        else:
            self.y_hat = np.concatenate([self.y_hat, y_hat], axis=0)

    def get_value(self):
        y_hat = np.argmax(self.y_hat, axis=-1)
        self.acc = (y_hat == self.y).mean()
        return self.acc

    def __str__(self):
        self.get_value()
        fmtstr = '{name} {acc' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)


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


class ProcessMonitor(object):
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
        fig_file_path = os.path.join('../Fig', fig_file)

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

    def update_validation(self, loss, acc):
        self.val_loss = loss
        self.val_acc = acc
        self.history['val_loss'].append(loss)
        self.history['val_accuracy'].append(acc)

        if self.val_acc > self.val_acc_best:
            self.best_str = 'val_accuracy improved from {0:.4f} to {1:.4f}, saving model to {2}'.format(self.val_acc_best, self.val_acc, self.model_file)
            self.val_acc_best = self.val_acc
        else:
            self.best_str = 'val_accuracy not improved {0:.4f}'.format(self.val_acc_best)

    def update_train(self, loss, acc, lr):
        self.loss = loss
        self.acc = acc
        self.history['loss'].append(loss)
        self.history['accuracy'].append(acc)

        self.lr = lr
        self.history['lr'].append(lr)


class SimSiamMonitor(object):
    def __init__(self, user=None):
        self.user = user
        # self.bone_file_name = user.bone_file_name
        #
        # self.log_file = self.bone_file_name + '_log.csv'
        # self.pt_csv_path = './pt_csv/'
        #
        # mkdir(self.pt_csv_path)
        #
        # self.model_file_blank = self.pt_csv_path + self.bone_file_name + '_%s_%d.pt'
        # self.history_file_blank = self.pt_csv_path + self.bone_file_name + '_%s_%d.csv'

        self.loss = 0.0
        self.lr = 0.0

        self.loss_best = 1.0e6

        self.history = {'loss': [], 'lr': []}

    def update(self, loss, lr):
        self.loss = loss
        self.history['loss'].append(loss)

        self.lr = lr
        self.history['lr'].append(lr)


class CoTeachingMonitor(object):
    def __init__(self, user=None):
        self.user = user

        self.history = {'acc1': [], 'loss1': [], 'acc2': [], 'loss2': [],
                        'val_acc1': [], 'val_acc2': []}

    def update(self, acc1, loss1, acc2, loss2, val_acc1, val_acc2):
        self.history['acc1'].append(acc1)
        self.history['loss1'].append(loss1)
        self.history['acc2'].append(acc2)
        self.history['loss2'].append(loss2)

        self.history['val_acc1'].append(val_acc1)
        self.history['val_acc2'].append(val_acc2)


class Monitor(object):
    def __init__(self, user=None):
        self.user = user
        self.history = {}
        self.key_any = None

    def update(self, **kwarg):
        for key in kwarg:
            self.key_any = key
            if key not in self.history:
                self.history[key] = [kwarg[key]]
            else:
                self.history[key].append(kwarg[key])

    def save(self, file_path):
        df_history = pd.DataFrame(self.history, index=[n for n in range(len(self.history[self.key_any]))])
        df_history.to_csv(file_path, index=False)


def mkdir(path):
    import os

    path = path.strip()         # 去除首尾空格
    path = path.rstrip("\\")    # 去除尾部 \ 符号

    if not os.path.exists(path):
        os.makedirs(path)
        return True
    else:
        return False



