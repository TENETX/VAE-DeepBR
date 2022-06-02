from scipy.io import loadmat, savemat
import numpy as np


dataname = r'data/education.mat'
dataset = loadmat(dataname)
case = 1

if case == 1:
    X = np.concatenate((dataset['train_data'], dataset['test_data']), axis=0)
    Y = np.concatenate((dataset['train_target'].T, dataset['test_target'].T), axis=0)

    dataset['data'] = X
    dataset['target'] = Y
else:
    dataset['data'] = dataset['dataset']
    dataset['target'] = dataset['class'].T
    dataset['target'] = (dataset['target'] > 0) * 1.0

savemat(dataname, mdict=dataset)
