import csv
import numpy as np
import math 
import random
from proj1_helpers import predict_labels

def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x, axis=0)
    x = x - mean_x
    std_x = np.std(x, axis=0)
    x = x / std_x
    return x, mean_x, std_x

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
 
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]
            
def split_data(x, y, ratio, seed=1):

    np.random.seed(seed)
    N = len(x)

    n_train = int(ratio * N)
    
    train_index = np.random.choice(N, n_train, replace=False)

    index = np.arange(N)

    mask = np.in1d(index, train_index)

    test_index = np.random.permutation(index[~mask])

    x_train = x[train_index]
    y_train = y[train_index]

    x_test = x[test_index]
    y_test = y[test_index]

    return x_train, y_train, x_test, y_test

def prediction(w_train,tx,y_test):
    
    y_pred1 = predict_labels(w_train,tx)

    for n,i in enumerate(y_test):
        if i==0:
              y_test[n]=-1
            
    right = 0
    wrong = 0

    for i in range(len(y_test)):
        if y_test[i] == y_pred1[i]:
            right +=1 
        else:
            wrong +=1 
    
        
    print("Good prediction : ", right)
    print("Bad predition : " , wrong)
    print("Ratio : " ,right/len(y_test))
    