import numpy as np
import scipy.io as sio
from torchvision import datasets, transforms

def load_EMNIST_data(file, verbose = False, standarized = False):
    """
    file should be the downloaded EMNIST file in .mat format.
    """
    mat = sio.loadmat(file)
    data = mat["dataset"]



    writer_ids_train = data['train'][0,0]['writers'][0,0]
    writer_ids_train = np.squeeze(writer_ids_train)
    X_train = data['train'][0,0]['images'][0,0]
    X_train = X_train.reshape((X_train.shape[0], 28, 28), order = "F")
    y_train = data['train'][0,0]['labels'][0,0]
    y_train = np.squeeze(y_train)
    y_train -= 1 #y_train is zero-based

    writer_ids_test = data['test'][0,0]['writers'][0,0]
    writer_ids_test = np.squeeze(writer_ids_test)
    X_test = data['test'][0,0]['images'][0,0]
    X_test= X_test.reshape((X_test.shape[0], 28, 28), order = "F")
    y_test = data['test'][0,0]['labels'][0,0]
    y_test = np.squeeze(y_test)
    y_test -= 1 #y_test is zero-based


    if standarized:
        X_train = X_train/255
        X_test = X_test/255
        mean_image = np.mean(X_train, axis=0)
        X_train -= mean_image
        X_test -= mean_image


    if verbose == True:
        print("EMNIST-letter dataset ... ")
        print("X_train shape :", X_train.shape)
        print("X_test shape :", X_test.shape)
        print("y_train shape :", y_train.shape)
        print("y_test shape :", y_test.shape)

    return X_train, y_train, X_test, y_test, writer_ids_train, writer_ids_test

def EMNIST_client_regenerate(data_train, label_train, writer_train, num_users):
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    for i in range(num_users):
        for j in range(20):
            temp = np.where(writer_train == i * 20 + j)
            dict_users[i] = np.concatenate((dict_users[i], temp[0][:]), axis=0)
    return dict_users

def ratio_loss_data(data_train, label_train, writer_train, num_class):
    dict_users = {i: np.array([], dtype='int64') for i in range(num_class)}
    for class_index in range(num_class):
        idx_temp = np.where(label_train == class_index)
        dict_users[class_index] = np.concatenate((dict_users[class_index], idx_temp[0][0:63]), axis=0)
    return dict_users

def EMNIST_client_imbalance(data_train, label_train, writer_train, num_users, minor_class, ratio):
    # dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    dict_raw = EMNIST_client_regenerate(data_train, label_train, writer_train, num_users)
    dict_users = dict_raw
    number_raw = np.zeros((1, 26))
    for i in range(num_users):
        for label in range(26):
            temp = 0
            for index in dict_raw[i]:
                if label_train[index] == label:
                    temp += 1
            number_raw[0, label] += temp
    for minor in minor_class:
        base_temp = int(number_raw[0, minor] * ratio)
        raw_temp = number_raw[0, minor]
        for i in range(num_users):
            for index in dict_raw[i]:
                if label_train[index] == minor:
                    dict_users[i] = np.delete(dict_users[i], np.where(dict_users[i] == index)[0])
                    raw_temp -= 1
                if raw_temp == base_temp:
                    break
    return dict_users

