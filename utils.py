# ------------------------------------------------------------------------------
# --coding='utf-8'--
# Written by wissing
# ------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import logging
import numpy as np
import configparser
from sklearn.metrics import auc


def read_config(ini_file):
    ''' Performs read config file and parses it.

    :param ini_file: (String) the path of a .ini file.
    :return config: (dict) the dictionary of information in ini_file.
    '''
    def _build_dict(items):
        return {item[0]: eval(item[1]) for item in items}
    # create configparser object
    cf = configparser.ConfigParser()
    # read .ini file
    cf.read(ini_file)
    config = {sec: _build_dict(cf.items(sec)) for sec in cf.sections()}
    return config

def Auc(pred, y):
    ''' Performs calculating c-index

    :param risk_pred: (np.ndarray or torch.Tensor) model prediction
    :param y: (np.ndarray or torch.Tensor) the times of event e
    :param e: (np.ndarray or torch.Tensor) flag that records whether the event occurs
    :return c_index: the c_index is calculated by (risk_pred, y, e)
    '''
    if not isinstance(y, np.ndarray):
        y = y.detach().cpu().numpy()
    if not isinstance(pred, np.ndarray):
        pred = pred.detach().cpu().numpy()
    return cal(y, pred)

def adjust_learning_rate(optimizer, epoch, lr, lr_decay_rate):
    ''' Adjusts learning rate according to (epoch, lr and lr_decay_rate)

    :param optimizer: (torch.optim object)
    :param epoch: (int)
    :param lr: (float) the initial learning rate
    :param lr_decay_rate: (float) learning rate decay rate
    :return lr_: (float) updated learning rate
    '''
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr / (1+epoch*lr_decay_rate)
    return optimizer.param_groups[0]['lr']

def create_logger(logs_dir):
    ''' Performs creating logger

    :param logs_dir: (String) the path of logs
    :return logger: (logging object)
    '''
    # logs settings
    log_file = os.path.join(logs_dir,
                            time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time())) + '.log')

    # initialize logger
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)

    # initialize handler
    handler = logging.FileHandler(log_file)
    handler.setLevel(logging.INFO)
    handler.setFormatter(
        logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

    # initialize console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)

    # builds logger
    logger.addHandler(handler)
    logger.addHandler(console)

    return logger


def cal(Y, P):
    Y = np.array(Y)
    P = np.array(P)
    #  print(Y, P)
    roc, roc_auc = ROCandAUC(P, Y)
    FPR, TPR, ACC, TP, FN, FP, TN = get_score(P, Y)

    return FPR, TPR, ACC, roc, roc_auc, TP, FN, FP, TN


# TPR
def get_score(SR, GT, threshold=0.5):
    SR_temp = SR.copy()
    SR_temp[SR_temp >= threshold] = 1.
    SR_temp[SR_temp < threshold] = 0.
    _TP = 0  # 正确肯定——实际是正例，识别为正例
    _FN = 0  # 错误否定（漏报）——实际是正例，却识别成了负例
    _FP = 0  # 错误肯定（误报）——实际是负例，却识别成了正例
    _TN = 0  # 正确否定——实际是负例，识别为负例
    for i in range(len(SR_temp)):
        # print(Y[i], P[i])
        if GT[i] == SR_temp[i]:
            if GT[i] == 1:
                _TP += 1
            elif GT[i] == 0:
                _TN += 1
        elif GT[i] != SR_temp[i]:
            if GT[i] == 1:
                _FN += 1
            elif GT[i] == 0:
                _FP += 1
    _FPR = _FP / (_FP + _TN)
    _TPR = _TP / (_TP + _FN)
    _ACC = (_TP + _TN) / (_TP + _FN + _FP + _TN)

    return _FPR, _TPR, _ACC, _TP, _FN, _FP, _TN


def ROCandAUC(SR, GT):
    ths = np.unique(SR)
    ths = np.sort(ths)
    # print(ths)
    roc = np.array([[0, 0]])
    for th in ths:
        fpr, tpr, _, _, _, _, _ = get_score(SR, GT, th)
        # print(tpr, fpr)
        roc = np.concatenate([roc, [[fpr, tpr]]], axis=0)
    roc = roc[1:]
    roc_auc = auc(roc[:, 0], roc[:, 1])
    return roc, roc_auc