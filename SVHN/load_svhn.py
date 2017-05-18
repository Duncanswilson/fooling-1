from __future__ import absolute_import
from __future__ import print_function

import os
import multiprocessing as mp
from subprocess import call
import warnings
import numpy as np
import scipy.io as sio
from keras.utils import to_categorical

def load_data_svhn(): 
    if not os.path.isfile("data/svhn_train.mat"):
        print('Downloading SVHN train set...')
        call(
            "curl -o data/svhn_train.mat "
            "http://ufldl.stanford.edu/housenumbers/train_32x32.mat",
            shell=True
        )
    if not os.path.isfile("data/svhn_test.mat"):
        print('Downloading SVHN test set...')
        call(
            "curl -o data/svhn_test.mat "
            "http://ufldl.stanford.edu/housenumbers/test_32x32.mat",
            shell=True
        )
    train = sio.loadmat('data/svhn_train.mat')
    test = sio.loadmat('data/svhn_test.mat')

    x_train = np.transpose(train['X'], axes=[3, 0, 1, 2])
    x_test = np.transpose(test['X'], axes=[3, 0, 1, 2])
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    # reshape (n_samples, 1) to (n_samples,) and change 1-index
    # to 0-index
    y_train = np.reshape(train['y'], (-1,)) 
    y_test = np.reshape(test['y'], (-1,))

    y_train[y_train == 10] = 0 
    y_test[y_test == 10] = 0 

    y_train_labels = y_train 
    y_test_labels = y_test

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    return x_train, y_train, x_test, y_test 
