import numpy as np
import torch
from scipy.interpolate import griddata
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import random
from torch.utils.data import Dataset, DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from utils import *

data_size = 12

def load_data():
    data_list = []
    for i in range(1, data_size+1):
        path = f"dataset/{i}/"
        X = np.load(path + "data.npy")
        Y = np.load(path + "label.npy")
        Y = Y.astype(np.int64)
        data_list.append((X, Y))
    return data_list

channel_str2idx = {v: k-1 for k, v in channelID2str.items()}

# change data into 2d
def get_CNN_data(X):
    n_sam, n_ch, dim = X.shape
    N, M = len(channel_matrix), len(channel_matrix[0])
    nX = np.zeros((n_sam, dim, N, M))

    for i, row in enumerate(channel_matrix):
        for j, ch in enumerate(row):
            if ch != '-' and ch in channel_str2idx:
                idx = channel_str2idx[ch]
                nX[:, :, i, j] = X[:, idx, :]

    return nX

# merge all the data except i as training set, and set i as training set
def merge_data(data_list, i):
    X_test, Y_test = data_list[i]
    train_data = [d for j, d in enumerate(data_list) if j != i]
    X_train = np.concatenate([d[0] for d in train_data], axis=0)
    Y_train = np.concatenate([d[1] for d in train_data], axis=0)

    n_train, n_ch, dim = X_train.shape
    n_test = X_test.shape[0]

    X_train_flat = X_train.reshape(n_train, -1)
    X_test_flat = X_test.reshape(n_test, -1)

    scaler = StandardScaler()
    X_train_flat = scaler.fit_transform(X_train_flat)
    X_test_flat = scaler.transform(X_test_flat)

    X_train = X_train_flat.reshape(n_train, n_ch, dim)
    X_test = X_test_flat.reshape(n_test, n_ch, dim)

    return X_train, Y_train, X_test, Y_test

def zscore(X):
    n_sam, n_ch, dim = X.shape
    X_flat = X.reshape(n_sam, -1)
    scaler = StandardScaler()
    X_flat = scaler.fit_transform(X_flat)
    return X_flat.reshape(n_sam, n_ch, dim)

def merge_data1(data_list, i):

    X_test, Y_test = data_list[i]
    train_data = [d for j, d in enumerate(data_list) if j != i]
    X_train = np.concatenate([d[0] for d in train_data], axis=0)
    Y_train = np.concatenate([d[1] for d in train_data], axis=0)

    return X_train, Y_train, X_test, Y_test

def augment_seed(X, noise=0.01, dropout=0.1, scale_range=(0.9, 1.1)):
    X_aug = X.copy()
    # scale = np.random.uniform(*scale_range)
    # X_aug *= scale

    # axis = ()
    # if X_aug.ndim == 3: axis = (1, 2)
    # else: axis = (1, 2, 3)

    # std = X_aug.std(axis=axis, keepdims=True) * noise
    # noise = np.random.randn(*X_aug.shape) * std
    # X_aug += noise

    if X_aug.ndim == 4:  # (N, C, H, W)
        mask = np.random.rand(*X_aug[:, :, 0, 0].shape) > dropout
        mask = mask[:, :, np.newaxis, np.newaxis]
    elif X_aug.ndim == 3:  # (N, 62, 5)
        mask = np.random.rand(*X_aug[:, :, 0].shape) > dropout
        mask = mask[:, :, np.newaxis]
    X_aug = X_aug * mask

    return X_aug

class SeedDataset(Dataset):
    def __init__(self, X, Y, augment=False):
        self.X = X
        self.Y = Y
        self.augment = augment
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        X_i = self.X[idx]
        Y_i = self.Y[idx]

        if self.augment:
            X_i = augment_seed(X_i[np.newaxis, ...])[0]

        X_i = torch.tensor(X_i, dtype=torch.float32)
        Y_i = torch.tensor(Y_i, dtype=torch.long)
        return X_i, Y_i
    
def time_window(data, T):
    X, Y = data
    
    X_new, Y_new = [], []
    start = 0

    for i in range(1, len(Y)):
        if Y[i] != Y[i-1]:
            segment = X[start:i]
            label = Y[start]
            for j in range(0, len(segment) - T + 1):
                X_new.append(segment[j:j+T])
                Y_new.append(label)
            start = i

    segment = X[start:]
    label = Y[start]
    for j in range(0, len(segment) - T + 1):
        X_new.append(segment[j:j+T])
        Y_new.append(label)
    
    X_new = np.array(X_new)
    X_new = X_new.reshape(X_new.shape[0], X_new.shape[1], -1)
    Y_new = np.array(Y_new)
    
    return (X_new, Y_new)

def get_time_window_list(data_list, T):
    data_list_new = []
    for data in data_list:
        data_list_new.append(time_window(data, T))
    return data_list_new

def get_domain(X, Y, i):
    mask = Y != i
    return X[mask], Y[mask]

class DANNDataset(Dataset):
    def __init__(self, X_all, Y_class, Y_domain):
        self.X_all = X_all
        self.Y_class = Y_class
        self.Y_domain = Y_domain
    
    def __len__(self):
        return len(self.X_all)
    
    def __getitem__(self, idx):
        X = self.X_all[idx]
        Y_c = self.Y_class[idx]
        Y_d = self.Y_domain[idx]

        X = torch.tensor(X, dtype=torch.float32)
        Y_c = torch.tensor(Y_c, dtype=torch.long)
        Y_d = torch.tensor(Y_d, dtype=torch.long)
        return X, Y_c, Y_d
