# ------------------------------------------------------------------------------
# --coding='utf-8'--
# Written by wissing
# ------------------------------------------------------------------------------
import h5py
import pandas as pd
from sklearn.utils import shuffle
import numpy as np
from sklearn.decomposition import PCA
import os


def make_data(type=0):
    train_df = None
    test_df = None

    # 按比例读取
    if type == 0:
        T_file = r'raw data/T.xlsx'
        T_df = pd.read_excel(T_file)
        T_df = shuffle(T_df)
        F_file = r'raw data/F.xlsx'
        F_df = pd.read_excel(F_file)
        F_df = shuffle(F_df)
        test_df = pd.concat([T_df[:17], F_df[:17]])
        test_df = shuffle(test_df)
        train_df = pd.concat([T_df[17:], F_df[17:]])
        train_df = shuffle(train_df)

    # 一起读取
    elif type == 1:
        all_file = r'raw data/all.xlsx'
        all_df = pd.read_excel(all_file)
        train_df = shuffle(all_df)
        test_df = train_df[:35]
        train_df = train_df[35:]

    # 分别读取
    elif type == 2:
        train_file = r'raw data/training.xlsx'
        test_file = r'raw data/validation.xlsx'
        train_df = pd.read_excel(train_file)
        test_df = pd.read_excel(test_file)

    test_p = test_df['name']
    test_e = test_df['label']
    test_x = test_df.drop(['label', 'name'], axis=1)

    train_p = train_df['name']
    train_e = train_df['label']
    train_x = train_df.drop(['label', 'name'], axis=1)

    if os.path.exists('data/20-6-11/ours.h5'):
        os.remove('data/20-6-11/ours.h5')
        ours_f = h5py.File('data/20-6-11/ours.h5')
        ours_f.create_dataset('train/x', data=train_x)
        ours_f.create_dataset('train/e', data=train_e)
        ours_f.create_dataset('test/x', data=test_x)
        ours_f.create_dataset('test/e', data=test_e)
        ours_f.create_dataset('train/p', data=train_p)
        ours_f.create_dataset('test/p', data=test_p)
        ours_f.keys()
        ours_f.close()
    else:
        ours_f = h5py.File('data/20-6-11/ours.h5')
        ours_f.create_dataset('train/x', data=train_x)
        ours_f.create_dataset('train/e', data=train_e)
        ours_f.create_dataset('test/x', data=test_x)
        ours_f.create_dataset('test/e', data=test_e)
        ours_f.create_dataset('train/p', data=train_p)
        ours_f.create_dataset('test/p', data=test_p)
        ours_f.keys()
        ours_f.close()


def merge_data():
    df_ball = pd.read_csv('./paper_raw_data/ball_dongda.csv')
    df_clinic = pd.read_csv('./paper_raw_data/clinic_dongda.csv')
    df_shell = pd.read_csv('./paper_raw_data/shell_dongda.csv')

    df_ball_done = pd.merge(df_ball, df_clinic, on='name')
    df_shell_done = pd.merge(df_shell, df_clinic, on='name')

    df_ball_done.set_index('name').to_csv('./paper_raw_data/ball.csv')
    df_shell_done.set_index('name').to_csv('./paper_raw_data/shell.csv')


def paper_data(name='ball', h5path='data/paper/ball.h5', mode='noClinic'):
    df_ball = pd.read_csv('./paper_raw_data/ball.csv')
    df_shell = pd.read_csv('./paper_raw_data/shell.csv')
    df = df_ball
    if name == 'shell':
        df = df_shell
    elif name == 'all':
        df_ball = df_ball.drop(['type', 'circled', 'liver', 'gender', 'age', 'time', 'Event'], axis=1)
        df = pd.merge(df_ball, df_shell, on='name')
        df.to_csv('paper_raw_data/all.csv')
    df['type'] = df['type'].replace('mutation', 1)
    df['type'] = df['type'].replace('wild', 0)
    # num = int(len(df) * 0.3)
    # index = np.random.randint(0, len(df), num, dtype=np.int)
    # np.savetxt('data/paper/index.txt', index)
    index = np.loadtxt('data/paper/index.txt', dtype=np.int)
    test_df = df.copy()
    train_df = df.copy()
    for i in range(len(df)):
        if i in index:
            train_df = train_df.drop([i], axis=0)
        else:
            test_df = test_df.drop([i], axis=0)

    test_y = test_df['type']
    print(test_df['name'])
    test_x = test_df.drop(['type', 'name', 'circled', 'liver', 'gender', 'age', 'time', 'Event'], axis=1)

    train_y = train_df['type']
    train_x = train_df.drop(['type', 'name', 'circled', 'liver', 'gender', 'age', 'time', 'Event'], axis=1)

    if mode == 'onlyClinic':
        test_y = test_df['type']
        test_x = pd.get_dummies(test_df[['circled', 'liver']])

        train_y = train_df['type']
        train_x = pd.get_dummies(train_df[['circled', 'liver']])

    elif mode == 'withClinic':
        test_y = test_df['type']
        one_hot = pd.get_dummies(test_df[['circled', 'liver']])
        test_x = test_df.drop(['type', 'name', 'circled', 'liver', 'gender', 'age', 'time', 'Event'], axis=1)
        test_x = test_x.join(one_hot)

        train_y = train_df['type']
        one_hot = pd.get_dummies(train_df[['circled', 'liver']])
        train_x = train_df.drop(['type', 'name', 'circled', 'liver', 'gender', 'age', 'time', 'Event'], axis=1)
        train_x = train_x.join(one_hot)

    print(len(test_df), len(train_df))

    if os.path.exists(h5path):
        os.remove(h5path)
        ours_f = h5py.File(h5path)
        ours_f.create_dataset('train/x', data=train_x)
        ours_f.create_dataset('train/y', data=train_y)
        ours_f.create_dataset('test/x', data=test_x)
        ours_f.create_dataset('test/y', data=test_y)
        ours_f.keys()
        ours_f.close()
    else:
        ours_f = h5py.File(h5path)
        ours_f.create_dataset('train/x', data=train_x)
        ours_f.create_dataset('train/y', data=train_y)
        ours_f.create_dataset('test/x', data=test_x)
        ours_f.create_dataset('test/y', data=test_y)
        ours_f.keys()
        ours_f.close()


def reindex(rate=0.3):
    df = pd.read_csv('./paper_raw_data/ball.csv')
    num = 35  # int(len(df) * rate)
    index = shuffle(np.arange(len(df)))[:num]
    # index = (np.arange(85, 96))
    np.savetxt('data/paper/index.txt', index)
    print(len(df), index)


# reindex()
paper_data('shell', 'data/paper/clinic.h5', 'onlyClinic')
paper_data('shell', 'data/paper/shell.h5', 'noClinic')
paper_data('ball', 'data/paper/ball.h5', 'noClinic')
paper_data('shell', 'data/paper/shell_clinic.h5', 'withClinic')
paper_data('balle', 'data/paper/ball_clinic.h5', 'withClinic')
paper_data('all', 'data/paper/all.h5', 'noClinic')
paper_data('all', 'data/paper/all_clinic.h5', 'withClinic')
