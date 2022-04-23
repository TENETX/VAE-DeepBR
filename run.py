# -*- coding: UTF-8 -*-
import os
import xlwt
from functools import reduce
import numpy as np
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
from config import Fold_numbers, loss_coefficient, Args
from models import Model
from train import train, test
from utills.common_tools import split_data_set_by_idx, ViewsDataset, init_random_seed, load_mat_data
from Vmodel import VAE_Encoder, VAE_Bernulli_Decoder


def run(args, save_dir, file_name, choose):
    choose_status = ['N', 'Only BR', 'VAE+BR combine', 'VAE+BR two_stage']
    print('*' * 30)
    print('dataset:\t', args.DATA_SET_NAME)
    print('optimizer:\t Adam')
    print('missing rate:\t', args.missing_rate)
    print('choose status:\t', choose_status[choose])
    print('smoothing rate:\t', args.smoothing_rate)
    print('*' * 30)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_name = save_dir + file_name

    features, labels, idx_list = load_mat_data(
        os.path.join(args.DATA_ROOT, args.DATA_SET_NAME + '.mat'), True)

    fold_list, metrics_results = [], []
    rets = np.zeros((Fold_numbers, 11))  # 11 metrics

    # VAE creates
    device = args.device
    n_feature = features.shape[1]
    n_label = labels.shape[1]

    args.Vnum_class = n_label
    args.Vdim_z = n_label

    for fold in range(Fold_numbers):
        TEST_SPLIT_INDEX = fold
        print('-' * 50 + '\n' + 'Fold: %s' % fold)

        # X, Y, Y_PRED, t_X, t_Y
        train_features, train_labels, train_partial_labels, test_features, test_labels = split_data_set_by_idx(
            {0: features}, labels, idx_list, TEST_SPLIT_INDEX, args)

        # load views features and labels
        views_dataset = ViewsDataset(
            train_features, train_partial_labels, device)
        views_data_loader = DataLoader(
            views_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        view_code_list = list(train_features.keys())
        view_feature_nums_list = [train_features[code].shape[1]
                                  for code in view_code_list]
        feature_dim = reduce(lambda x, y: x + y, view_feature_nums_list)
        label_nums = train_labels.shape[1]

        # load model
        model = Model(feature_dim, label_nums, device, args).to(device)
        if choose != 1:
            enc = VAE_Encoder(n_in=n_feature + n_label,
                              n_hidden=args.Vn_hidden, n_out=args.Vdim_z, keep_prob=args.Vkeep_prob)
            dec = VAE_Bernulli_Decoder(n_in=args.Vdim_z, n_hidden=args.Vn_hidden,
                                       n_out=n_feature, keep_prob=args.Vkeep_prob)
        else:
            enc = None
            dec = None
        # training
        loss_list = train(model, device, views_data_loader, args, loss_coefficient,
                          train_features, train_partial_labels, test_features, test_labels, choose, enc, dec, fold=1)
        metrics_results, _ = test(
            model, test_features, test_labels, device, is_eval=True, args=args)
        fold_list.append(loss_list)
        for i, key in enumerate(metrics_results):
            rets[fold][i] = metrics_results[key]

    print("\n------------BR--summary--------------")
    means = np.mean(rets, axis=0)
    stds = np.std(rets, axis=0)
    metrics_list = list(metrics_results.keys())
    workbook = xlwt.Workbook(encoding='utf-8')
    sheet = workbook.add_sheet("savesheet")
    for i, _ in enumerate(means):
        sheet.write(i, 0, metrics_list[i])
        a = str('%.4f' % means[i])
        b = str('%.4f' % stds[i])
        sheet.write(i, 1, a + "±" + b)
        print("{metric}\t{means:.4f}±{std:.4f}".format(
            metric=metrics_list[i], means=means[i], std=stds[i]))
    # workbook.save(save_name + ".xls")
    # writer.flush()
    # writer.close()


if __name__ == '__main__':
    args = Args()

    # setting random seeds
    init_random_seed(args.seed)
    args.device = 'cpu'

    missing_rates = [0.3]

    datanames = ['genbase']

    # 1：BR, 2: VAE+BR, 3: Two_stage
    chooses = [3]
    for choose in chooses:
        for dataname in datanames:
            for p in missing_rates:
                args.DATA_SET_NAME = dataname
                args.missing_rate = p
                save_dir = f'results/{dataname}/'
                save_name = f'{args.DATA_SET_NAME}-p{p}-r{args.missing_num}'
                run(args, save_dir, save_name, choose)
