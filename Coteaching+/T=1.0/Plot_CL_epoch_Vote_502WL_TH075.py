
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import os

import logging

"""
小四：12
五：  10.5
小五： 9
六号： 7.5
小六： 6.5
七号： 5.5
八号： 5
"""
label_font = {'family': 'Times New Roman', 'size': 9}
ticks_font = {'family': 'Times New Roman', 'size': 5.5}
mark_font = {'family': 'Times New Roman', 'size': 3}
text_font = {'family': 'Times New Roman', 'size': 5}


def single_fif_plot(df_epochs, dy=0.0):
    """
    单独plot一个fif的epochs, 便于用颜色区分.
    :param df_epochs:
    :param s:
    :param dy:
    :return:
    """
    df = df_epochs

    plt.plot(df[['SC']][df['label'] == 0] + dy, 'dimgray')
    plt.plot(df[['SC']][df['label'] == 1] + dy)


def score_plot(files=None, th=0.5, info_file=None, phase=1):

    # df_info = pd.read_csv(info_file, low_memory=False)
    #
    # bl_list = df_info[df_info['Black_List']].index.to_list()

    # plt.figure('CL', figsize=(16, 9))
    # plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=None, hspace=None)

    df_buffer = None
    sn = 0
    dy = 0
    for file in files[-1::-1]:

        if not os.path.exists(file):
            if file == '':
                dy += 0.5
            print(file)
            continue

        df = pd.read_csv(file, low_memory=False)
        fif_index_list = df[df['ID'] == 0]['fif_idx'].to_list()

        df.loc[df[df['Black_List']].index, 'SC'] = None

        df['SC'] = df[['SC1', 'SC2']].mean(axis=1)

        if df_buffer is None:
            df_buffer = df.copy()
            df_buffer['BL_Sel'] = True

        df_buffer['SC%d' % (sn, )] = df['SC']
        sn += 1

        #############################################

        # 垂直分割线
        n_epochs_of_fif = df[['upper_limit']][df['ID'] == 0].to_numpy().flatten()
        fif_ticks = np.cumsum(n_epochs_of_fif)
        plt.vlines(0, dy, dy + 1.0, colors='gray', linestyles='--',  linewidth=0.25)
        for x in fif_ticks:
            plt.vlines(x, dy, dy + 1.0, colors='gray', linestyles='--', linewidth=0.25)

        #############################################

        # plt.plot(df[['y_star']] + dy)
        plt.hlines(0.5 + dy, 0, len(df), colors='k', linestyles='--', linewidth=0.25)
        # plt.text(0, dy+1.1, file)

        # 分fif显示
        for idx in set(df['fif_idx'].tolist()):
            df_epochs_of_fif = df[df['fif_idx'] == idx].copy()

            # single_fif_plot(df_epochs_of_fif, dy)
            plt.plot(df_epochs_of_fif[['SC']][df_epochs_of_fif['label'] == 0] + dy, 'dimgray', linewidth=0.25)
            plt.plot(df_epochs_of_fif[['SC']][df_epochs_of_fif['label'] == 1] + dy, linewidth=0.25)

        plt.text(-1500, dy+0.5, 'SC%d' % (sn, ),
                 ha='left', va='center',
                 color='k', fontdict=text_font)

        #############################################

        # 黑名单选择方法
        # bl_sel0 = ((entropy < entropy_mean) & (score_mean < 0.5))
        # bl_sel1 = (scores.max(axis=1) < 0.5)
        bl_sel = df['SC'] < 0.5

        # bl_sel = df_buffer['label'] != df_buffer['y_star']

        plt.plot(bl_sel*0.4 + dy - 1.0, 'k', linewidth=0.25)

        plt.text(-1500, dy-1.0, 'BL%d' % (sn, ),
                 ha='left', va='bottom',
                 color='k', fontdict=text_font)

        # 计算黑epochs的比例
        th = th
        df['BL_Sel'] = bl_sel
        df_buffer['BL_Sel'] = df_buffer['BL_Sel'] & bl_sel
        bl0_list = []
        bl1_list = []
        for index in fif_index_list:#":#df_info.index:

            bl_i = df[df['fif_idx'] == index]['BL_Sel']
            label_i = df[df['fif_idx'] == index]['label'].any()
            # print(label_i)
            rate_bl = bl_i.mean()
            n_bl = bl_i.sum()
            n_epochs = len(bl_i)

            x = fif_ticks[index] - n_epochs_of_fif[index]/2

            y = -0.5 + dy

            s_rate = '%.2f' % (rate_bl, )
            s_info = '%d/%d' % (n_bl, n_epochs)

            if rate_bl > th:
                # plt.text(x, y, s_rate, ha='center', color='r', fontdict=mark_font, rotation=45)
                plt.plot(x, y+0.1, 'v', color='r', markersize=0.5)
                if label_i == 0:
                    bl0_list.append(index)
                else:
                    bl1_list.append(index)

            #     plt.text(x, y - 0.75, '%d' % (index, ),
            #              ha='center', color='b', fontdict=mark_font, rotation=45)
            # else:
            #     if rate_bl > 0.1:
            #         plt.text(x, y, s_rate,
            #                  ha='center', color='b', fontdict=mark_font, rotation=45)

            # plt.text(x, y-0.2, s_info, ha='center', color='gray')

        # plt.text(0, dy+1.5, file + ' Black_List_%s: %d/%d' % (str(th), len(bl0_list), len(bl1_list)))
        # plt.text(0, dy+1.1, '%s / %s' % (str(bl0_list), str(bl1_list)))

        # for copy
        print(file, 'Black_List_%s: %d/%d' % (str(th), len(bl0_list), len(bl1_list)))
        print(bl0_list)
        print(bl1_list)

        logging.info(file + 'Black_List_%s: %d/%d' % (str(th), len(bl0_list), len(bl1_list)))
        logging.info(bl0_list)
        logging.info(bl1_list)
        logging.info('\n')

        dy += 2.5

    #############################################

    # Mean
    # dy -= 1
    sc_cols = ['SC%d' % (n, ) for n in range(sn)]
    df_buffer['SC'] = df_buffer[sc_cols].mean(axis=1)
    # plt.plot(df_buffer['SC'] + dy, 'dimgray')

    # 表决, 黑名单选择方法
    df = df_buffer
    # df['BL_Sel'] = df['SC'] < 0.5

    # bl_sel = df_buffer['label'] != df_buffer['y_star']

    plt.plot(df['BL_Sel'] * 0.4 + dy - 1.0, 'b',
             label='Phase %d' % (phase, ),
             linewidth=0.25)
    plt.legend(shadow=True, prop=ticks_font)

    plt.text(-1500, dy-1.0, 'BL',
             ha='left', va='bottom',
             color='b', fontdict=text_font)

    # 计算黑epochs的比例
    th = th
    # df['BL_Sel'] = bl_sel
    bl0_list = []
    bl1_list = []
    for index in fif_index_list:  # ":#df_info.index:

        bl_i = df[df['fif_idx'] == index]['BL_Sel']
        label_i = df[df['fif_idx'] == index]['label'].any()
        # print(label_i)
        rate_bl = bl_i.mean()
        n_bl = bl_i.sum()
        n_epochs = len(bl_i)

        x = fif_ticks[index] - n_epochs_of_fif[index] / 2

        y = -0.5 + dy

        s_rate = '%.2f' % (rate_bl,)
        s_info = '%d/%d' % (n_bl, n_epochs)

        if rate_bl > th:
            # plt.text(x, y, s_rate, ha='center', color='r', fontdict=mark_font, rotation=45)
            plt.plot(x, y+0.1, 'v', color='r', markersize=0.5)
            if label_i == 0:
                bl0_list.append(index)
            else:
                bl1_list.append(index)

        #     plt.text(x, y - 0.75, '%d' % (index, ), ha='center', color='b', fontdict=mark_font, rotation=45)
        # else:
        #     if rate_bl > 0.1:
        #         plt.text(x, y, s_rate, ha='center', color='b', fontdict=mark_font, rotation=45)

        # plt.text(x, y-0.2, s_info, ha='center', color='gray')

    plt.text(0, dy + .35, 'Blacklist by Vote: %d/%d' % (len(bl0_list), len(bl1_list)), fontdict=ticks_font)
    # plt.text(0, dy + .0, '%s / %s' % (str(bl0_list), str(bl1_list)), fontdict=ticks_font)

    # for copy
    print(file, 'Vote Black List_%s: %d/%d' % (str(th), len(bl0_list), len(bl1_list)))
    print(bl0_list)
    print(bl1_list)
    dy += 3

    logging.info('Black List by Vote _%s: %d/%d' % (str(th), len(bl0_list), len(bl1_list)))
    logging.info(bl0_list)
    logging.info(bl1_list)
    logging.info('\n')

    #
    # # Entropy
    # entropy_disp(df_buffer, sn, dy)

    #############################################

    # plt.grid(axis='y', which='both')
    # plt.yticks(np.arange(0, dy, 0.5))
    plt.ylim((-1.5, dy-1.5))
    plt.yticks([])

    n_epochs = df[['upper_limit']][df['ID'] == 0].to_numpy().flatten()
    ticks = np.cumsum(n_epochs) - n_epochs/2    # xtick locations
    # print(n_epochs.shape)

    fif_idx = df[['fif_idx']][df['ID'] == 0].to_numpy().flatten()
    # print(fif_idx.shape)

    # plt.xticks(font=ticks_font['family'], fontsize=ticks_font['size'])
    plt.xticks(ticks[::10], fif_idx[::10],
               font=ticks_font['family'], fontsize=ticks_font['size'])
    # plt.xticks(rotation=-45)

    return bl0_list, bl1_list


def entropy_disp(df_buffer, n_cols, dy):
    sc_cols = ['SC%d' % (n, ) for n in range(n_cols)]

    # 计算Entropy
    p = df_buffer[sc_cols].to_numpy()
    entropy = -p * np.log(p, where=p != 0)  # 分别计算p*log（p）
    df_buffer['Entropy'] = entropy.mean(axis=1)

    plt.plot(df_buffer['Entropy'] + dy - 0.5, 'gray')
    plt.hlines(df_buffer['Entropy'].mean() + dy - 0.5, 0, len(df_buffer), colors='k', linestyles='--', linewidth=1.0)

    df_buffer['WL_SEL'] = df_buffer['Entropy'] < df_buffer['Entropy'].mean()
    plt.plot(df_buffer['WL_SEL'] * 0.25 + dy - 1.5, 'g')

    # plt.figure('Scatter Entropy', figsize=(6, 6))
    # plt.scatter(df_buffer['Mean'], df_buffer['Entropy'], s=1)
    # plt.hlines(df_buffer['Entropy'].mean(), 0., 1., 'r', linestyles='--')
    # # plt.hlines(df['Entropy'].median(), 0., 1., 'r', linestyles='-.')
    # plt.show()


def evaluate_df_epochs_with_CL(df_examples):
    """

    :return:
    """
    from numpy import nan

    # 计算各个类别self-confidence的均值, 作为阈值
    t0 = df_examples[df_examples['label'] == 0]['SC'].mean()
    t1 = df_examples[df_examples['label'] == 1]['SC'].mean()

    df_examples['Prob1'] = (df_examples['SC'] + df_examples['label'] - 1).abs()
    df_examples['Prob0'] = (df_examples['label'] - df_examples['SC']).abs()

    # 候选预测输出
    df_examples['y_0'] = df_examples['Prob0'] >= t0
    df_examples['y_1'] = df_examples['Prob1'] >= t1

    df_examples['y_star'] = nan

    df_y_0 = df_examples[df_examples['y_0']]
    df_examples.loc[df_y_0.index, 'y_star'] = 0

    df_y_1 = df_examples[df_examples['y_1']]
    df_examples.loc[df_y_1.index, 'y_star'] = 1  # 可能collision

    # 消除存在的collision
    df_y_collision = df_examples[df_examples['y_0'] & df_examples['y_1']]
    y_collision_pred = df_y_collision[['Prob0', 'Prob1']].to_numpy().argmax(axis=1)
    df_examples.loc[df_y_collision.index, 'y_star'] = y_collision_pred

    return df_examples


def register_logging():
    py_code_file = os.path.split(__file__)[-1]  # 获取代码文件名 *.py

    logging.basicConfig(filename=py_code_file + '.log', level=logging.INFO,
                        format='%(asctime)s %(message)s')

    logging.info('\n')


if __name__ == '__main__':

    register_logging()

    plt.figure('CL', figsize=(5.5, 3.0))
    plt.subplots_adjust(left=0.05, bottom=0.075, right=0.95, top=0.975, wspace=None, hspace=None)

    score_plot(files=[

        # 'EEGLab_CoT+_FC1_BLA__D512_TC16V3_IMB2__C502_CTL-CM_EC_epochs_0.csv',
        # 'EEGLab_CoT+_FC1_BLA__D512_TC16V3_IMB2__C502_CTL-CM_EC_epochs_1.csv',
        # 'EEGLab_CoT+_FC1_BLA__D512_TC16V3_IMB2__C502_CTL-CM_EC_epochs_2.csv',
        # 'EEGLab_CoT+_FC1_BLA__D512_TC16V3_IMB2__C502_CTL-CM_EC_epochs_3.csv',
        # 'EEGLab_CoT+_FC1_BLA__D512_TC16V3_IMB2__C502_CTL-CM_EC_epochs_4.csv',

        # 'EEGLab_CoT+_FC1_BLA__D512_TC16V3_IMB2__C502_P2_CTL-CM_EC_epochs_0.csv',
        # 'EEGLab_CoT+_FC1_BLA__D512_TC16V3_IMB2__C502_P2_CTL-CM_EC_epochs_1.csv',
        # 'EEGLab_CoT+_FC1_BLA__D512_TC16V3_IMB2__C502_P2_CTL-CM_EC_epochs_2.csv',
        # 'EEGLab_CoT+_FC1_BLA__D512_TC16V3_IMB2__C502_P2_CTL-CM_EC_epochs_3.csv',
        # 'EEGLab_CoT+_FC1_BLA__D512_TC16V3_IMB2__C502_P2_CTL-CM_EC_epochs_4.csv',

        'EEGLab_CoT+_FC1_BLA__D512_TC16V3_IMB2__C502_P3_CTL-CM_EC_epochs_0.csv',
        'EEGLab_CoT+_FC1_BLA__D512_TC16V3_IMB2__C502_P3_CTL-CM_EC_epochs_1.csv',
        'EEGLab_CoT+_FC1_BLA__D512_TC16V3_IMB2__C502_P3_CTL-CM_EC_epochs_2.csv',
        'EEGLab_CoT+_FC1_BLA__D512_TC16V3_IMB2__C502_P3_CTL-CM_EC_epochs_3.csv',
        'EEGLab_CoT+_FC1_BLA__D512_TC16V3_IMB2__C502_P3_CTL-CM_EC_epochs_4.csv',

        # 'EEGLab_CoT+_FC1__D512_TC16V3_IMB2__C502_P4_CTL-CM_EC_epochs_0.csv',
        # 'EEGLab_CoT+_FC1__D512_TC16V3_IMB2__C502_P4_CTL-CM_EC_epochs_1.csv',
        # 'EEGLab_CoT+_FC1__D512_TC16V3_IMB2__C502_P4_CTL-CM_EC_epochs_2.csv',
        # 'EEGLab_CoT+_FC1__D512_TC16V3_IMB2__C502_P4_CTL-CM_EC_epochs_3.csv',
        # 'EEGLab_CoT+_FC1__D512_TC16V3_IMB2__C502_P4_CTL-CM_EC_epochs_4.csv',

        # 'EEGLab_CleanLab_FC1_BAL__D512_TC16V3_IMB2_CTL-CM_EC_epochs_0.csv',
        # 'EEGLab_CleanLab_FC1_BAL__D512_TC16V3_IMB2_CTL-CM_EC_epochs_1.csv',
        # 'EEGLab_CleanLab_FC1_BAL__D512_TC16V3_IMB2_CTL-CM_EC_epochs_2.csv',
        # 'EEGLab_CleanLab_FC1_BAL__D512_TC16V3_IMB2_CTL-CM_EC_epochs_3.csv',
        # 'EEGLab_CleanLab_FC1_BAL__D512_TC16V3_IMB2_CTL-CM_EC_epochs_4.csv',

    ], th=0.75, phase=3)

    # plt.title('Vote Phase I')
    plt.savefig('Vote_Phase_3.png',
                # transparent=True,   # 图片背景透明
                dpi=600
                )

    plt.show()




