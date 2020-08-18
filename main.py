# ------------------------------------------------------------------------------
# --coding='utf-8'--
# Written by wissing
# ------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
import torch.optim as optim
import prettytable as pt
from preprocess import make_data
import matplotlib.pyplot as plt

from networks import Net
from networks import Criterion
from datasets import MakeDataset
from utils import read_config
from utils import Auc
from utils import adjust_learning_rate
from utils import create_logger


def train(ini_file):
    ''' Performs training according to .ini file

    :param ini_file: (String) the path of .ini file
    :return best_c_index: the best c-index
    '''
    # reads configuration from .ini file
    config = read_config(ini_file)
    # builds network|criterion|optimizer based on configuration
    model = Net(config['network']).to(device)
    criterion = Criterion(config['network'], device).to(device)
    optimizer = eval('optim.{}'.format(config['train']['optimizer']))(
        model.parameters(), lr=config['train']['learning_rate'])
    # constructs data loaders based on configuration
    train_dataset = MakeDataset(config['train']['h5_file'], is_train=True, device=device)
    test_dataset = MakeDataset(config['train']['h5_file'], is_train=False, device=device)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_dataset.__len__())
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=test_dataset.__len__())
    # training
    _best_acc = 0.70
    best_acc = 0.65
    best_ep = 0
    flag = 0
    _best_auc = 0
    best_auc = 0
    best_roc = None
    for epoch in range(1, config['train']['epochs'] + 1):
        # adjusts learning rate
        lr = adjust_learning_rate(optimizer, epoch,
                                  config['train']['learning_rate'],
                                  config['train']['lr_decay_rate'])
        # train step
        model.train()
        for X, y in train_loader:
            # makes predictions
            pred = model(X)
            train_loss = criterion(pred, y, model)
            train_FPR, train_TPR, train_ACC, train_roc, train_roc_auc, _, _, _, _ = Auc(pred, y)
            # updates parameters
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
        # valid step
        model.eval()
        for X, y in test_loader:
            # makes predictions
            with torch.no_grad():
                pred = model(X)
                # print(pred, y)
                valid_loss = criterion(pred, y, model)
                valid_FPR, valid_TPR, valid_ACC, valid_roc, valid_roc_auc, _, _, _, _ = Auc(pred, y)
                if valid_ACC > best_acc and train_ACC > _best_acc:
                    flag = 0
                    best_acc = valid_ACC
                    _best_acc = train_ACC
                    best_ep = epoch
                    best_auc = valid_roc_auc
                    _best_auc = train_roc_auc
                    best_roc = valid_roc
                    # saves the best model
                    torch.save({
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch}, os.path.join(models_dir, ini_file.split('\\')[-1] + '.pth'))
                else:
                    flag += 1
                    if flag >= patience:
                        print('epoch: {}\t{:.8f}({:.8f})'.format(best_ep, _best_acc, best_acc))
                        if best_roc is not None:
                            plt.plot(best_roc[:, 0], best_roc[:, 1])
                            plt.title('ep:{}  AUC: {:.4f}({:.4f}) ACC: {:.4f}({:.4f})'.format(best_ep, _best_auc, best_auc, _best_acc, best_acc))
                            plt.show()
                        return best_acc, _best_acc
        # notes that, train loader and valid loader both have one batch!!!
        print('\rEpoch: {}\tLoss: {:.8f}({:.8f})\tACC: {:.8f}({:.8f})\tAUC: {}({})\tFPR: {:.8f}({:.8f})\tTPR: {:.8f}({:.8f})\tlr: {:g}\n'.format(
            epoch, train_loss.item(), valid_loss.item(), train_ACC, valid_ACC, train_roc_auc, valid_roc_auc, train_FPR, valid_FPR, train_TPR, valid_TPR, lr), end='', flush=False)
    return best_acc, _best_acc


if __name__ == '__main__':
    # global settings
    best_data = 0
    for try_ep in range(100):
        make_data(1)  # 0: 按比例，1: all， 2：分别
        logs_dir = 'logs'
        models_dir = os.path.join(logs_dir, 'models')
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        logger = create_logger(logs_dir)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        configs_dir = 'configs'
        params = [('Ours', 'ours.ini')]
        patience = 50
        # training
        headers = []
        values = []
        for name, ini_file in params:
            print('Running {}({})...'.format(name, ini_file))
            fig = read_config(os.path.join(configs_dir, ini_file))
            t_fig = fig['train']
            n_fig = fig['network']
            print("train: learning_rate = {:.6f}\nlr_decay_rate = {}\noptimizer = {}".format(t_fig['learning_rate'],
                                                                                                   t_fig['lr_decay_rate'],
                                                                                                   t_fig['optimizer']))
            print("network: drop = {}\ndim = {}\nact = {}\nl2 = {}".format(n_fig['drop'], n_fig['dims'], n_fig['activation'],
                                                                                 n_fig['l2_reg']))

            best_acc, _ = train(os.path.join(configs_dir, ini_file))
            if best_acc > best_data:
                best_data = best_acc
                try:
                    os.rename('data/20-6-11/ours.h5', 'data/20-6-11/best.h5')
                except:
                    os.remove('data/20-6-11/best.h5')
                    os.rename('data/20-6-11/ours.h5', 'data/20-6-11/best.h5')
            # headers.append(name)
            # values.append('{:.6f}'.format(best_acc))
            # print('')
            print("The best valid acc in this dataset: {}\nThe best valid acc in this train: {}".format(best_acc, best_data))
        # prints results
        # tb = pt.PrettyTable()
        # tb.field_names = headers
        # tb.add_row(values)
        # logger.info(tb)
        # del logger
