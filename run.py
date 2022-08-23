# -*- coding: UTF-8 -*-
import os
import time

import torch
import numpy as np
from torch.utils.data import DataLoader

from config import *
from mpvae import VAE
from train import test, Trainer
from utils.common_tools import split_data_set_by_idx, load_mat_data, init_random_seed, Dataset, save_results


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
    # if os.path.exists(save_name):
    #     return

    features, labels, idx_list = load_mat_data(os.path.join(args.DATA_ROOT, args.DATA_SET_NAME + '.mat'), fold_num=Fold_numbers,
                                               need_zscore=True)

    fold_list, metrics_results = [], []
    rets = np.zeros([Fold_numbers, 15])   # last col: time
    start = time.time()
    for fold in range(Fold_numbers):
        TEST_SPLIT_INDEX = fold
        print('-' * 50 + '\n' + 'Fold: %s' % fold)
        train_features, train_labels, test_features, test_labels = split_data_set_by_idx(
            features, labels, idx_list, TEST_SPLIT_INDEX, device)
        
        # load features and labels
        train_dataset = Dataset(train_features, train_labels, device)
        train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=False, num_workers=0)

        # building the model
        args.feature_dim = train_features.shape[1]
        args.label_dim = train_labels.shape[1]
        model = VAE(args).to(device)

        # training
        # print(model)
        trainer = Trainer(model, args, device)
        loss_list = trainer.fit(train_data_loader, train_features, train_labels, test_features, test_labels)

        fold_list.append(loss_list)
        metrics_results, _ = test(model, test_features, test_labels, None, device, is_eval=True)

        for i, key in enumerate(metrics_results):
            rets[fold][i] = metrics_results[key]

    time_usage = time.time() - start
    save_results(save_name, metrics_results, fold_list, time_usage, args)


if __name__ == '__main__':
    args = Args()

    device = torch.device("cuda") if args.cuda else torch.device("cpu")

    # lrs = [1e-2, 5e-2, 2e-3, 6e-3, 5e-3, 1e-4, 5e-4, 1e-5, 1e-6]

    # datanames = ['Emotions', 'Scene', 'Yeast', 'Pascal', 'Iaprtc12', 'Corelq', 'Mirflickr', 'Espgame']
    # label_nums = [6, 6, 14, 20, 291, 260, 38, 268]

    # datanames = ['emotions']
    # datanames = ['scene']
    # datanames = ['Yeast']
    # datanames = ['Pascal']
    # datanames = ['Iaprtc12']
    # datanames = ['Corel5k']
    # datanames = ['Espgame']
    # datanames = ['sider_data']
    # datanames = ['fish_data']

    datanames = ['mirflickr_data']

    # param_grid = {
    #     'latent_dim': [6],
    #     'high_feature_dim': [256],
    #     'embedding_dim': [256],
    # }

    for i, dataname in enumerate(datanames):
        args.DATA_SET_NAME = dataname
        save_dir = f'results/{args.DATA_SET_NAME}/'
        save_name = f'{args.DATA_SET_NAME}-1.txt'
        run(device, args, save_dir, save_name)

            # Random Grid Search
            # random_params = {k: random.sample(v, 1)[0] for k, v in param_grid.items()}
            # args.latent_dim = random_params['latent_dim']
            # args.high_feature_dim = random_params['high_feature_dim']
            # args.embedding_dim = random_params['embedding_dim']
            # args.common_embedding_dim = random_params['common_embedding_dim']

