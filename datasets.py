# ------------------------------------------------------------------------------
# --coding='utf-8'--
# Written by wissing
# ------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import h5py
import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset


class MakeDataset(Dataset):
    ''' The dataset class performs loading data from .h5 file. '''

    def __init__(self, h5_file, is_train, device):
        ''' Loading data from .h5 file based on (h5_file, is_train).

        :param h5_file: (String) the path of .h5 file
        :param is_train: (bool) which kind of data to be loaded?
                is_train=True: loading train data
                is_train=False: loading test data
        '''
        # loads data
        self.X, self.y = \
            self._read_h5_file(h5_file, is_train)
        # normalizes data
        self._normalize()

        self.device = device

        print('=> load {} samples'.format(self.X.shape[0]))

    def _read_h5_file(self, h5_file, is_train):
        ''' The function to parsing data from .h5 file.

        :return X: (np.array) (n, m)
            m is features dimension.
        :return e: (np.array) (n, 1)
            whether the event occurs? (1: occurs; 0: others)
        :return y: (np.array) (n, 1)
            the time of event e.
        '''
        split = 'train' if is_train else 'test'
        with h5py.File(h5_file, 'r') as f:
            X = f[split]['x'][()]
            y = f[split]['y'][()].reshape(-1, 1)
        return X, y

    def _normalize(self):
        ''' Performs normalizing X data. '''
        for i in range(len(self.X)):
            if (self.X[i].max() - self.X[i].min()) == 0:
                continue
            self.X[i] = (self.X[i] - self.X[i].min()) / (self.X[i].max() - self.X[i].min())

    def __getitem__(self, item):
        ''' Performs constructing torch.Tensor object'''
        # gets data with index of item
        X_item = self.X[item].astype(np.float32)  # (m)
        y_item = self.y[item].astype(np.float32)  # (1)
        # constructs torch.Tensor object
        X_tensor = torch.from_numpy(X_item).to(self.device)
        y_tensor = torch.from_numpy(y_item).to(self.device)  # .long()
        return X_tensor, y_tensor

    def __len__(self):
        return self.X.shape[0]
