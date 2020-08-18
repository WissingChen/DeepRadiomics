# ------------------------------------------------------------------------------
# --coding='utf-8'--
# Written by wissing
# ------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
import numpy as np
import torch.optim as optim
import prettytable as pt
import matplotlib.pyplot as plt

from networks import Net
from networks import Criterion
from datasets import MakeDataset
from utils import read_config
from utils import Auc
from utils import adjust_learning_rate
import pandas as pd


class ANN(object):

    def __init__(self, data: str):
        self.logs_dir = 'logs'
        self.models_dir = os.path.join(self.logs_dir, 'models')
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.configs_dir = 'configs'
        self.name = data
        self.ini_file = '%s.ini' % data
        self.patience = 200

    def test(self):
        ''' Performs training according to .ini file

            :param ini_file: (String) the path of .ini file
            :return best_c_index: the best c-index
            '''
        # reads configuration from .ini file
        ini_file = os.path.join(self.configs_dir, self.ini_file)
        config = read_config(ini_file)
        # builds network|criterion|optimizer based on configuration
        model = Net(config['network']).to(self.device)
        # constructs data loaders based on configuration
        train_dataset = MakeDataset(config['train']['h5_file'], is_train=True, device=self.device)
        test_dataset = MakeDataset(config['train']['h5_file'], is_train=False, device=self.device)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=train_dataset.__len__())
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=test_dataset.__len__())
        model.load_state_dict(torch.load(os.path.join(self.models_dir, ini_file.split('\\')[-1] + '-ANN.pth'))['model'])
        # test step
        model.train(False)
        for X, y in train_loader:
            # makes predictions
            with torch.no_grad():
                pred = model(X)
                pred = pred.detach().cpu().numpy()
                y = y.detach().cpu().numpy()
                valid_FPR, valid_TPR, valid_ACC, valid_roc, valid_roc_auc, _, _, _, _ = Auc(pred, y)
                print(valid_roc_auc)
                np.savetxt('pred.txt', pred, fmt='%.6f')
                np.savetxt('label.txt', y, fmt='%d')
                # (pred-0.5)*model.semantic_output.weight
        # print(model.state_dict().keys())
        # w = model.state_dict()['model.0.weight'].detach().cpu().numpy()
        # b = model.state_dict()['model.0.bias'].detach().cpu().numpy()
        # print(w.T.shape, b.shape)



A = ANN('_ball_clinic')
A.test()


def test():
    data = pd.read_excel('paper_raw_data/prediction_train.xlsx')
    y = data['label']
    pred = data['ball with predict']
    valid_FPR, valid_TPR, valid_ACC, valid_roc, valid_roc_auc, _, _, _, _ = Auc(np.array(pred), np.array(y))
    print(valid_roc_auc)

# test()