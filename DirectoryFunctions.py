import numpy as np
import math

# Helper functions
# Author: Tomasz Hachaj
# e-mail: tomekhachaj@o2.pl
# 2020

def return_indexes(data, feature_id, percent_of_dataset):
    data_c1_index = np.where(data[:, feature_id] == 0)[0]
    data_c2_index = np.where(data[:, feature_id] == 1)[0]

    data_c1_len = len(data_c1_index)
    data_c2_len = len(data_c2_index)

    if data_c1_len > data_c2_len:
        train_l = math.floor(data_c2_len * percent_of_dataset)
    else:
        train_l = math.floor(data_c1_len * percent_of_dataset)

    data_c1_index_train = np.copy(data_c1_index[0:train_l])
    data_c2_index_train = np.copy(data_c2_index[0:train_l])

    data_c1_index_valid = np.copy(data_c1_index[train_l:data_c1_len])
    data_c2_index_valid = np.copy(data_c2_index[train_l:data_c2_len])

    data_train = np.concatenate([data_c1_index_train, data_c2_index_train])
    data_valid = np.concatenate([data_c1_index_valid, data_c2_index_valid])

    perm = np.arange(data_train.shape[0])
    np.random.shuffle(perm)
    data_train = data_train[perm]

    perm = np.arange(data_valid.shape[0])
    np.random.shuffle(perm)
    data_valid = data_valid[perm]

    return [data_train, data_valid]

def append_to_file(file_name, text):
    f = open(file_name, "a")
    f.write(text)
    f.flush()
    f.close()

def append_line_to_file(file_name, text):
    append_to_file(file_name, text + '\n')