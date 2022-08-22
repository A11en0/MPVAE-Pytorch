# -*- coding: UTF-8 -*-
import os
import random

from scipy.io import loadmat
import numpy as np
from sklearn import preprocessing
from torch.utils.data import Dataset
import torch
import h5py

from utils.random_noise import random_noise, random_noise_p_r


class Dataset(Dataset):
    def __init__(self, features, labels, device='cpu'):
        self.features = features
        self.labels = labels
        self.device = device
        self.features = features

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, item):
        return self.features[item], self.labels[item], item

def gen_idx_list(length, fold):
    idxs = np.random.permutation(length)
    idx = []
    fold_size = length // fold
    for i in range(fold):
        if i == fold - 1:
            idx.append(list(idxs[i * fold_size:]))
        else:
            idx.append(list(idxs[i*fold_size:(i+1)*fold_size]))

    return idx

def load_mat_data(file_name, fold_num=5, need_zscore=False):
    try:
        dataset = loadmat(file_name)
        if type(dataset.get('target')) == np.ndarray:
            features = dataset['data']
            target = dataset['target'].T
        else:
            features = dataset['dataset']
            target = dataset['class'].T
    except:
        dataset = loadmat(file_name)
        train_features = dataset['train_data']
        test_features = dataset['test_data']
        train_target = dataset['train_target'].T
        test_target = dataset['test_target'].T
        features = np.concatenate([train_features, test_features], axis=0)
        target = np.concatenate([train_target, test_target], axis=0)

    target = np.array(target, dtype=np.float32)
    target = torch.from_numpy(target)
    target = (target > 0)*1.0
    idx_list = gen_idx_list(target.shape[0], fold_num)
    features = np.array(features, dtype=np.float32)

    if need_zscore:
        features = preprocessing.scale(features)

    features = torch.from_numpy(features)
    return features, target, idx_list

def split_data_set_by_idx(features, labels, idx_list, test_split_id, device):
    train_idx_list = []
    test_idx_list = idx_list[test_split_id]
    for i in range(len(idx_list)):
        if i != test_split_id:
            train_idx_list.extend(idx_list[i])
    train_idx_list = [i.astype(dtype=np.int64) for i in train_idx_list]
    test_idx_list = [i.astype(dtype=np.int64) for i in test_idx_list]
    train_labels = labels[train_idx_list].to(device)
    test_labels = labels[test_idx_list].to(device)
    train_features = features[train_idx_list].to(device)
    test_features = features[test_idx_list].to(device)

    return train_features, train_labels, test_features, test_labels


# read data with matlab format by loadmat
def read_mat_data(file_name, need_zscore=True):
    # since the data is small, we load all data to the memory
    data = loadmat(file_name)
    features = data['data']
    views_count = features.shape[0]
    views_features = {}
    for i in range(views_count):
        view_feature = features[i][0]
        # change sparse to dense
        if type(view_feature) != type(np.array([1])):
            view_feature = view_feature.toarray()
        view_feature = np.array(view_feature, dtype=np.float32)
        if need_zscore:
            view_feature = preprocessing.scale(view_feature)
        # views_features['view_' + str(i)] = view_feature
        views_features[i] = torch.from_numpy(view_feature)
    labels = data['target']
    if type(labels) != type(np.array(1)):
        labels = labels.toarray()
    labels = np.array(labels, dtype=np.float32)
    labels = torch.from_numpy(labels)
    return views_features, labels

def init_random_seed(seed):
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

