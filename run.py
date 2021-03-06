# -*- coding: UTF-8 -*-
import os
from functools import reduce

import torch
import numpy as np
from torch.utils.data import DataLoader

from config import *
from mpvae import VAE
from train import test, Trainer
from utils.common_tools import split_data_set_by_idx, ViewsDataset, load_mat_data, init_random_seed


def run(device, args, save_dir, file_name):
    print('*' * 30)
    print('seed:\t', args.seed)
    print('dataset:\t', args.DATA_SET_NAME)
    print('latent dim:\t', args.latent_dim)
    print('optimizer:\t Adam')
    print('*' * 30)

    # setting random seeds
    init_random_seed(args.seed)

    save_name = save_dir + file_name

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if os.path.exists(save_name):
        return

    features, labels, idx_list = load_mat_data(os.path.join(args.DATA_ROOT, args.DATA_SET_NAME + '.mat'), fold_num=Fold_numbers,
                                               need_zscore=True)

    fold_list, metrics_results = [], []
    for fold in range(Fold_numbers):
        TEST_SPLIT_INDEX = fold
        print('-' * 50 + '\n' + 'Fold: %s' % fold)
        train_features, train_labels, train_partial_labels, test_features, test_labels = split_data_set_by_idx(
            features, labels, idx_list, TEST_SPLIT_INDEX, args)
        
        # load views features and labels
        views_dataset = ViewsDataset(train_features, train_partial_labels, device)
        views_data_loader = DataLoader(views_dataset, batch_size=256, shuffle=True,
                                       num_workers=args.workers, pin_memory=True,
                                       drop_last=False)

        view_code_list = list(train_features.keys())
        view_feature_dim_list = [train_features[code].shape[1] for code in view_code_list]
        feature_dim = reduce(lambda x, y: x + y, view_feature_dim_list)

        # get feature_dim and label_dim
        args.feature_dim = feature_dim
        args.label_dim = train_labels.shape[1]

        # building the model
        model = VAE(args).to(device)

        # training
        print(model)
        trainer = Trainer(model, args, device)
        loss_list = trainer.fit(views_data_loader, train_features, train_partial_labels, test_features, test_labels, fold)

        fold_list.append(loss_list)

        metrics_results, _ = test(model, test_features, test_labels, None, device, is_eval=True)

        # for i, key in enumerate(metrics_results):
        #     rets[fold][i] = metrics_results[key]

    rets = np.zeros((Fold_numbers, args.epochs, 11))  # 11 metrics
    for fold, li_fold in enumerate(fold_list):
        for i, epoch in enumerate(li_fold):
            for j, key in enumerate(li_fold[i]):
                rets[fold][i][j] = li_fold[i][key]

    means = np.mean(rets, axis=0)
    stds = np.std(rets, axis=0)
    _index = np.argsort(means[:, 6])

    mean_choose = means[_index][-1, :]
    std_choose = stds[_index][-1, :]

    print("\n------------summary--------------")
    print("Best Epoch: ", _index[-1])
    metrics_list = list(metrics_results.keys())

    with open(save_name, "w") as f:
        for i, _ in enumerate(mean_choose):
            print("{metric}\t{means:.4f}??{std:.4f}".format(metric=metrics_list[i], means=mean_choose[i],
                                                           std=std_choose[i]))
            f.write("{metric}\t{means:.4f}??{std:.4f}".format(metric=metrics_list[i], means=mean_choose[i],
                                                           std=std_choose[i]))
            f.write("\n")
        f.write(str(_index[-1]))


if __name__ == '__main__':
    args = Args()

    device = torch.device("cuda") if args.cuda else torch.device("cpu")
    # device = 'cpu'

    # lrs = [1e-2, 5e-2, 2e-3, 6e-3, 5e-3, 1e-4, 5e-4, 1e-5, 1e-6]
    # noise_rates = [0.3, 0.5, 0.7]
    noise_rates = [0.3]

    # datanames = ['Emotions', 'Scene', 'Yeast', 'Pascal', 'Iaprtc12', 'Corelq', 'Mirflickr', 'Espgame']
    # label_nums = [6, 6, 14, 20, 291, 260, 38, 268]

    # datanames = ['Emotions']
    datanames = ['Scene']
    # datanames = ['Yeast']
    # datanames = ['Pascal']
    # datanames = ['Iaprtc12']
    # datanames = ['Corel5k']
    # datanames = ['Mirflickr']
    # datanames = ['Espgame']

    # param_grid = {
    #     'latent_dim': [6],
    #     'high_feature_dim': [256],
    #     'embedding_dim': [256],
    # }

    for i, dataname in enumerate(datanames):
        for p in noise_rates:
            args.DATA_SET_NAME = dataname
            args.noise_rate = p

            save_dir = f'results/{args.DATA_SET_NAME}/'
            save_name = f'{args.DATA_SET_NAME}-p{args.noise_rate}.txt'
            run(device, args, save_dir, save_name)

            # Random Grid Search
            # random_params = {k: random.sample(v, 1)[0] for k, v in param_grid.items()}
            # args.latent_dim = random_params['latent_dim']
            # args.high_feature_dim = random_params['high_feature_dim']
            # args.embedding_dim = random_params['embedding_dim']
            # args.common_embedding_dim = random_params['common_embedding_dim']

