import numpy as np
import torch
import random
from torch.utils.data import Dataset
from scipy.io import loadmat
from utills.random_noise import missing_label


def setup_seed(seed):
    # 设置随机种子
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def mll_rec_loss(preds, targets, eps=1e-12):
    w1 = 1 / targets.sum(1)
    loss = -targets * (torch.log(preds + eps))
    loss = loss.sum(1) * w1
    return loss.mean(0)


def gauss_kl_loss(mu, sigma, eps=1e-12):
    mu_square = torch.pow(mu, 2)
    sigma_square = torch.pow(sigma, 2)
    loss = mu_square + sigma_square - torch.log(eps + sigma_square) - 1
    loss = 0.5 * loss.mean(1)
    return loss.mean()


def t_softmax(d, t=1):
    for i in range(len(d)):
        d[i] = d[i] * t
        d[i] = np.exp(d[i]) / sum(np.exp(d[i]))
    return d


class MLLDataset(Dataset):
    def __init__(self, args):
        super(MLLDataset, self).__init__()
        datas = loadmat(args['src_path'])
        self.n_feature = datas['data'].shape[1]
        self.n_label = datas['target'].shape[0]
        self.sparse = False
        self.genDataSets(datas)

    def genDataSets(self, datas):
        n_sample = datas['target'].shape[1]
        feature_data = datas['data']
        label_data = datas['target'].T
        label_data, num = missing_label(label_data, 0.3, 3)
        print(feature_data[0][0].shape)
        print(feature_data[0][1].shape)
        print(label_data.shape)
        print(self.n_feature)
        print(self.n_label)
        print(n_sample)
        features = torch.zeros(n_sample, self.n_feature)
        labels = torch.zeros(n_sample, self.n_label)
        if self.sparse:
            for i in range(n_sample):
                feature = torch.from_numpy(
                    np.array(feature_data[i], dtype=np.int64))
                label = torch.from_numpy(
                    np.array(label_data[i], dtype=np.int64))
                if len(feature) > 0:
                    features[i].scatter_(0, feature, 1)
                if len(label) > 0:
                    labels[i].scatter_(0, label, 1)
        else:
            for i in range(n_sample):
                feature = torch.from_numpy(np.array(feature_data[i]))
                label = torch.from_numpy(np.array(label_data[i]))
                features[i] = feature
                labels[i] = label
            # Normalization
            max_data = features.max()
            min_data = features.min()
            features = (features - min_data) / (max_data - min_data)
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]
