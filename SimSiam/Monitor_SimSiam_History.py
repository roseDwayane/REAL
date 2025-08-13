
from matplotlib import pyplot as plt
import pandas as pd

import keyboard


label_font = {'family': 'Times New Roman', 'size': 7.5}   # 9
ticks_font = {'family': 'Times New Roman', 'size': 5}   # 7.5
legend_font = {'family': 'Times New Roman', 'size': 5}


def history_plot(files=None):
    """
    可以plot多个文件.
    :param files:
    :return:
    """
    plt.subplots_adjust(left=0.1, bottom=0.15, right=0.9, top=0.95,
                        wspace=None, hspace=0.25)

    n_files = len(files)
    for m, file in enumerate(files):
        df = pd.read_csv(file)

        # epochs = df.shape[0]
        # up_to_date_acc = df['val_accuracy'].to_numpy()[-1]

        # 右图
        ax1 = plt.subplot(n_files, 1, m+1)
        # fig, ax1 = plt.subplots(1, 2)

        line1 = plt.plot(df[['lr']], '--', linewidth=0.5, color='gray', label='LR')
        plt.ylabel('Learning rate', fontdict=label_font, labelpad=2)
        plt.yticks(font=ticks_font['family'], fontsize=ticks_font['size'])

        plt.xlabel('Epochs', fontdict=label_font, labelpad=2)
        plt.xticks(font=ticks_font['family'], fontsize=ticks_font['size'])

        # plt.legend(loc='upper left', shadow=False, prop=legend_font, frameon=True)

        ax2 = ax1.twinx()

        line2 = plt.plot(df[['loss']], '-g', linewidth=0.5, label='Loss')
        pos_min = df['loss'].argmin()
        line3 = plt.plot(pos_min, df['loss'].min(), 'or',
                         markersize=2.0,
                         label='Loss = %.4f@%d' % (df['loss'].min(), df['loss'].argmin()))

        # plt.legend(loc='upper right', shadow=False, prop=legend_font, frameon=True)

        # plt.xlim([0, 200])
        plt.ylim([-1.025, -0.25])
        # plt.title(file)
        plt.ylabel('Training loss', fontdict=label_font, labelpad=2)
        plt.yticks(font=ticks_font['family'], fontsize=ticks_font['size'])

        # 合并图例
        lines = line1 + line2 + line3
        labels = [line.get_label() for line in lines]
        plt.legend(lines, labels, shadow=False, prop=legend_font, frameon=True)

        plt.savefig('Z_Hist_Simsiam.png', dpi=600)


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
                                             initialdir='./pt_csv')
    print(full_names)
    return full_names


if __name__ == '__main__':

    history_files = get_file_name()

    # 显示
    plt.figure('history', figsize=(3.5, min(2.0, 2.0*len(history_files))))

    plt.ion()

    while True:

        history_plot(files=history_files)

        plt.pause(30)

        plt.clf()
        plt.ioff()

        """plt.pause(50)

        plt.figure('acc')
        plt.clf()

        plt.figure('loss')
        plt.clf()"""




