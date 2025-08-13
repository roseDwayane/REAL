
from matplotlib import pyplot as plt
import pandas as pd

import keyboard


def history_plot(files=None):
    """
    可以plot多个文件.
    :param files:
    :return:
    """

    plt.rcParams.update({'font.size': 16})
    plt.legend(loc='upper right')

    plt.subplots_adjust(left=0.1, bottom=0.05, right=0.9, top=0.95,
                        wspace=0.25, hspace=0.25)

    n_files = len(files)
    for m, file in enumerate(files):
        df = pd.read_csv(file)
        history_len = df.shape[0]

        # 右图
        ax1 = plt.subplot(n_files, 1, m+1)

        # plt.plot(df[['lr']])
        plt.ylabel('LR')

        ax2 = ax1.twinx()

        plt.plot(df[['accuracy']], label='acc')
        plt.plot(df[['val_accuracy']], label='val_acc')

        pos = history_len - 1 - df['val_accuracy'][-1::-1].argmax()     # val_acc >= best_val_acc
        # pos = df['val_accuracy'].argmax()                               # val_acc > best_val_acc
        plt.plot(pos, df['val_accuracy'].max(), 'o')

        plt.ylim([0.0, 1.01])
        plt.ylabel('Acc')

        plt.plot(df[['loss']], label='loss')
        plt.plot(df[['val_loss']], label='val_loss')

        plt.legend()

        # # 最佳验证精度时的Loss
        # x = min(10, df[['lr']].shape[0]/2)
        # plt.text(x, 0.6, 'Loss=%.4f' % (df['loss'].min()))


flag = False


def key_press(key):
    global flag
    if key.name == 'space':
        flag = True


keyboard.on_press(key_press)


def get_file_name():
    from tkinter import filedialog
    import tkinter as tk

    # 创建打开文件窗口
    fold = tk.Tk()
    fold.withdraw()

    # 可以打开多个文件
    full_names = filedialog.askopenfilenames(filetypes=[("Keras History File(*.csv)", "*.csv"), ("All File(*.*)", "*.*")],
                                             initialdir='./h5_csv')
    print(full_names)
    return full_names


if __name__ == '__main__':

    history_files = get_file_name()

    # 显示
    plt.figure('history', figsize=(16, min(9, 5*len(history_files))))

    plt.ion()

    while True:

        history_plot(files=history_files)

        plt.pause(30)

        # for n in range(30):
        #     plt.pause(1)
        #     if flag:
        #         exit(30)

        plt.clf()
        plt.ioff()

        """plt.pause(50)

        plt.figure('acc')
        plt.clf()

        plt.figure('loss')
        plt.clf()"""




