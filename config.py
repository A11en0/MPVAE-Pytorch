# -*- coding: UTF-8 -*-

Fold_numbers = 5
TEST_SPLIT_INDEX = 1


class Args:
    def __init__(self):
        self.DATA_ROOT = './datasets'
        self.DATA_SET_NAME = 'Emotions'
        self.epochs = 100
        self.show_epoch = 1
        self.model_save_epoch = 20
        self.model_save_dir = 'model_save_dir'
        self.using_lp = False

        self.neighbors_num = 10
        self.no_verbose = True
        self.using_lp = False  # label propagation for label de-noise

        self.is_test_in_train = True
        self.batch_size = 512
        self.seed = 43
        self.cuda = True
        self.workers = 0
        self.opt = 'adam'
        self.lr = 1e-3  # 1e-3 5e-3 6e-4
        self.weight_decay = 1e-5  # 1e-5
        self.noise_rate = 0.7
        self.noise_num = 3

        self.mode = 'train'
        self.residue_sigma = 'random'
        self.feature_dim = 294
        self.label_dim = 6

        self.latent_dim = 256
        self.z_dim = 100
        self.n_train_sample = 256
        self.nll_coeff = 0.5
        self.c_coeff = 0.5

        self.keep_prob = 0.5
        self.scale_coeff = 1.0
