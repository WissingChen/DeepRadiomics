# ------------------------------------------------------------------------------
# --coding='utf-8'--
# Written by wissing
# ------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn


class Regularization(object):
    def __init__(self, order, weight_decay):
        ''' The initialization of Regularization class

        :param order: (int) norm order number
        :param weight_decay: (float) weight decay rate
        '''
        super(Regularization, self).__init__()
        self.order = order
        self.weight_decay = weight_decay

    def __call__(self, model):
        ''' Performs calculates regularization(self.order) loss for model.

        :param model: (torch.nn.Module object)
        :return reg_loss: (torch.Tensor) the regularization(self.order) loss
        '''
        reg_loss = 0
        for name, w in model.named_parameters():
            if 'weight' in name:
                reg_loss = reg_loss + torch.norm(w, p=self.order)
        reg_loss = self.weight_decay * reg_loss
        return reg_loss


class Net(nn.Module):
    ''' The module class performs building network according to config'''

    def __init__(self, config):
        super(Net, self).__init__()
        # parses parameters of network from configuration
        self.drop = config['drop']
        self.norm = config['norm']
        self.dims = config['dims']
        self.activation = config['activation']
        # builds network
        self.model = self._build_network()
        self.semantic = nn.Linear(4, 1)
        self.output = nn.Linear(self.dims[-1], 1)
        self.semantic_output = nn.Linear(2, 1)

    def _build_network(self):
        ''' Performs building networks according to parameters'''
        layers = []
        for i in range(len(self.dims) - 1):
            if i and self.drop is not None:  # adds dropout layer
                layers.append(nn.Dropout(self.drop))
            # adds linear layer
            layers.append(nn.Linear(self.dims[i], self.dims[i + 1]))
            if self.norm:  # adds batchnormalize layer
                layers.append(nn.BatchNorm1d(self.dims[i + 1]))
            # adds activation layerz
            layers.append(eval('nn.{}()'.format(self.activation)))
        # builds sequential network
        return nn.Sequential(*layers)

    def forward(self, X):
        # sp op
        _X = X[:, -4:]
        X = X[:, :-4]
        _x = self.semantic(_X)
        x = self.model(X)
        x = self.output(x)
        x = torch.cat([x, _x], dim=1)
        x = self.semantic_output(x).sigmoid()
        # x = nn.functional.softmax(self.output(x), dim=1)
        # x = (self.output(x)).sigmoid()
        return x


class Criterion(nn.Module):
    def __init__(self, config, device):
        super(Criterion, self).__init__()
        self.L2_reg = config['l2_reg']
        self.reg = Regularization(order=2, weight_decay=self.L2_reg)
        self.CEL = nn.CrossEntropyLoss()
        self.MSE = nn.MSELoss()
        self.device = device

    def forward(self, pred, y, model):
        # loss = self.CEL(pred, y.squeeze())
        loss = self.MSE(pred, y)
        l2_loss = self.reg(model)
        return loss + l2_loss
