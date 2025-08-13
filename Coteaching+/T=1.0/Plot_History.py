
from matplotlib import pyplot as plt
import pandas as pd

import keyboard

"""
小四：12
五：  10.5
小五： 9
六号： 7.5
小六： 6.5
七号： 5.5
八号： 5
"""

# label_font = {'family': 'Times New Roman', 'size': 9}
# ticks_font = {'family': 'Times New Roman', 'size': 5.5}
# mark_font = {'family': 'Times New Roman', 'size': 3}
# text_font = {'family': 'Times New Roman', 'size': 5}

label_font = {'family': 'Times New Roman', 'size': 9}   # 六号
ticks_font = {'family': 'Times New Roman', 'size': 5}
text_font = {'family': 'Times New Roman', 'size': 2}
legend_font = {'family': 'Times New Roman', 'size': 5}

sn_font = {'family': 'Times New Roman', 'size': 9}   # 小五


def history_plot(files=None, sub_sn=1):
    """
    可以plot多个文件.
    :param files:
    :return:
    """
    # plt.subplots_adjust(left=0.1, bottom=0.05, right=0.9, top=0.95,
    #                     wspace=0.25, hspace=0.25)

    n_files = len(files)
    for m, file in enumerate(files):
        df = pd.read_csv(file)

        # 右图
        # plt.subplot(n_files, 2, m*2+2)
        # ax1 = plt.subplot(n_files, 2, m*2+2)    # 双纵坐标
        # # fig, ax1 = plt.subplots(1, 2)

        # 显示
        plt.figure('Acc',
                   figsize=(2.5, min(2.0, 1.5 * len(files))),
                   dpi=600)
        plt.subplots_adjust(left=0.15, bottom=0.2, right=0.95, top=0.95,
                            wspace=0.2, hspace=None)

        plt.plot(df[['acc1', 'acc2']], label=['Acc$_{A}$', 'Acc$_{B}$'], linewidth=0.5)
        # plt.plot(df[['val_acc1', 'val_acc2']], label=['val_acc1', 'val_acc2'])

        plt.legend(shadow=False, prop=legend_font, frameon=True)

        plt.xticks(font=ticks_font['family'], fontsize=ticks_font['size'])
        plt.yticks(font=ticks_font['family'], fontsize=ticks_font['size'])

        # # 子图序号
        # plt.text(100.0, 0.576, '(b)',   # 1
        # # plt.text(100.0, 0.758, '(b)',   # 2
        # # plt.text(100.0, 0.776, '(b)',   # 3
        #          va='bottom', ha='center',
        #          fontsize=sn_font['size'])

        # ax2 = ax1.twinx()                     # 双纵坐标

        # 参考线
        # plt.plot([0, epochs], [up_to_date_acc, up_to_date_acc], '--b', linewidth=0.5)

        # plt.xlim([0, 200])
        # plt.ylim([-1.0, 0])
        # plt.ylim([0.45, 1.05])
        # plt.title('Acc')
        plt.ylabel('Accuracy', font=label_font['family'], fontsize=label_font['size'], labelpad=0)
        plt.xlabel('Epochs', font=label_font['family'], fontsize=label_font['size'], labelpad=0)

        plt.savefig('Z_Acc_CoT+_%d.png' % (sub_sn),     # 需调整子图序号位置
                    # transparent=True,         # 图片背景透明
                    dpi=600
                    )

        # 左图
        # plt.subplot(n_files, 2, m*2+1)

        # plt.plot(df[['loss1', 'loss2']])
        # 显示
        plt.figure('Loss',
                   figsize=(2.5, min(2.0, 1.5 * len(files))),
                   dpi=600)
        plt.subplots_adjust(left=0.15, bottom=0.2, right=0.95, top=0.95,
                            wspace=0.2, hspace=None)

        try:
            plt.plot(df['lossA'],  label='Loss$_{A}$', linewidth=0.5)
            plt.plot(df['lossB'], label='Loss$_{B}$', linewidth=0.5)
            plt.legend(shadow=False, prop=legend_font, frameon=True)
        except Exception as e:
            print(e)

        # plt.ylim([-0.05, 1.05])
        # plt.title('Loss')
        plt.ylabel('Loss', font=label_font['family'], fontsize=label_font['size'], labelpad=0)
        plt.xlabel('Epochs', font=label_font['family'], fontsize=label_font['size'], labelpad=0)

        plt.xticks(font=ticks_font['family'], fontsize=ticks_font['size'])
        plt.yticks(font=ticks_font['family'], fontsize=ticks_font['size'])

        # # 子图序号
        # plt.text(100.0, -0.4, '(a)',
        #          va='bottom', ha='center',
        #          fontsize=sn_font['size'])

        plt.savefig('Z_Loss_CoT+_%d.png' % (sub_sn),     # 需调整子图序号位置
                    # transparent=True,         # 图片背景透明
                    dpi=600
                    )


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

    # # 显示
    # plt.figure('history',
    #            figsize=(5.5, min(2.0, 2.0*len(history_files))),
    #            dpi=600)
    # plt.subplots_adjust(left=0.075, bottom=0.25, right=0.975, top=0.95,
    #                     wspace=0.2, hspace=None)

    # plt.ion()

    while True:
        # history_plot(file='h5_csv/EEGLab_Classify_100_CTL-CM_EEGNet_v100b_3.csv')

        history_plot(files=history_files, sub_sn=3)

        # plt.savefig('Z_History_CoT+_1.png',     # 需调整子图序号位置
        #             # transparent=True,         # 图片背景透明
        #             dpi=600
        #             )

        break

        plt.pause(300)

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




