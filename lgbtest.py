import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
import json
import random
import os
import warnings
import argparse
import re
import seaborn as sns
pd.options.mode.chained_assignment = None
warnings.filterwarnings('ignore')

np.random.seed(0)
random.seed(0)
os.environ["PYTHONHASHSEED"] = "0"

with open('lgb１.json', 'r') as file:
    lgbjson = json.load(file)
    datapath = lgbjson["load_dataset"]["args"]["path"]["data"]
    preprocess = lgbjson["preprocess"]["args"]["type"]["data"]
    if preprocess == "StandardScaler":
        sc = StandardScaler()
    parm = lgbjson["create_LightGBM"]['args']
    min_lr = parm["lr"]["data"][1]
    max_lr = parm["lr"]["data"][0]
    object = parm["objective"]["data"][0]
    metric = parm["metric"]["data"][0]
    seed = parm["seed"]["data"]
    min_max_depth = parm["max_depth"]["data"][0]
    max_max_depth = parm["max_depth"]["data"][1]
    min_min_data_in_leaf = parm["min_data_in_leaf"]["data"][0]
    max_min_data_in_leaf = parm["min_data_in_leaf"]["data"][1]
    min_num_leaf = parm["num_leaves"]["data"][0]
    max_num_leaf = parm["num_leaves"]["data"][1]
    hpo = lgbjson["train"]["args"]["search"]["data"][0]

# *** split into free and anomaly ***
def free_anomaly_split(X, Y):
    free = []
    anomaly = []

    for x, y in zip(X, Y):
        if y == 0:
            free.append(x)
        elif y == 1:
            anomaly.append(x)

    free = np.array(free)
    anomaly = np.array(anomaly)

    return free, anomaly

def smooth_curve(x):
    #x=1 dimension array
    window_len = 11
    s = np.r_[x[window_len-1:0:-1], x, x[-1:-window_len:-1]]
    w = np.kaiser(window_len, 2)
    y = np.convolve(w/w.sum(), s, mode='valid')
    return y[5:len(y)-5]


# create_dataset function
def create_dataset(dataset, look_back):
    data_X = np.zeros((len(dataset) - look_back + 1, 3))
    j = 0

    for i in range(look_back - 1, len(dataset)):
        data_pre = dataset[i - look_back + 1:i + 1, 0]

        data_pre_mean = np.mean(data_pre, axis=0)
        data_pre_min = np.min(data_pre, axis=0)
        data_pre_max = np.max(data_pre, axis=0)

        data_X[j, :] = np.array([data_pre_mean, data_pre_min, data_pre_max])
        j += 1

    return np.array(data_X).reshape(-1, 3)


def lgb_train_predict(x_train, y_train, x_valid, y_valid, x_test, y_test, params, \
                      test_flag=False):
    lgb_train = lgb.Dataset(x_train, y_train)
    lgb_valid = lgb.Dataset(x_valid, y_valid)
    lgb_test = lgb.Dataset(x_test, y_test)

    model_lgb = lgb.train(params=params, train_set=lgb_train, \
                          valid_sets=[lgb_train, lgb_valid], \
                          verbose_eval=0, early_stopping_rounds=20)

    if test_flag:
        test_pred = np.zeros((len(y_test), 1))
        test_pred[:, 0] = np.where(model_lgb.predict(x_test) >= 0.5, 1, 0)
        test_acc = accuracy_score(y_test.reshape(-1, 1), test_pred)
        test_f1score = f1_score(y_test.reshape(-1, 1), test_pred)
        test_cm = confusion_matrix(y_test.reshape(-1, 1), test_pred)

        return test_acc, test_f1score, test_cm, test_pred, model_lgb

    else:
        train_pred = np.zeros((len(y_train), 1))
        train_pred[:, 0] = np.where(model_lgb.predict(x_train) >= 0.5, 1, 0)
        train_acc = accuracy_score(y_train.reshape(-1, 1), train_pred)

        valid_pred = np.zeros((len(y_valid), 1))
        valid_pred[:, 0] = np.where(model_lgb.predict(x_valid) >= 0.5, 1, 0)
        valid_acc = accuracy_score(y_valid.reshape(-1, 1), valid_pred)

        return train_acc, valid_acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--HPO", default="random")
    args = parser.parse_args()
    if args.HPO == "random":
        hypo = hpo

    valve1_data = pd.read_csv(datapath)
    print(valve1_data)
    train_pre=valve1_data

    train_pre_size = len(train_pre)
    train_size = int(train_pre_size*0.7)
    train = train_pre[0:train_size]
    x_train_pre = train.drop('anomaly', axis = 1)
    x_train = x_train_pre.drop('changepoint', axis = 1)
    y_train= train['anomaly'].values

    valid_pre_size = train_pre_size - train_size
    valid_size = int(valid_pre_size*0.66)
    valid = train_pre[train_size:train_size + valid_size]
    x_valid_pre = valid.drop('anomaly', axis=1)
    x_valid = x_valid_pre.drop('changepoint', axis=1)
    y_valid = valid['anomaly'].values

    test=train_pre[train_size+valid_size:]
    x_test_pre=test.drop('anomaly',axis=1)
    x_test=x_test_pre.drop('changepoint',axis=1)
    y_test=test['anomaly'].values

    x_train_win=np.zeros_like(x_train.values)
    x_valid_win=np.zeros_like(x_valid.values)
    x_test_win=np.zeros_like(x_test.values)

    data_dim = 8
    for i in range(0, data_dim):
        x_train_win[:, i] = smooth_curve(x_train.values[:, i].flatten())
        x_valid_win[:, i] = smooth_curve(x_valid.values[:, i].flatten())
        x_test_win[:, i] = smooth_curve(x_test.values[:, i].flatten())

    x_train_std = sc.fit_transform(x_train_win)
    x_valid_std = sc.transform(x_valid_win)
    x_test_std = sc.transform(x_test_win)

    look_back = 10
    data_dim = 8

    for i in range(0, data_dim):

        if i == 0:
            # train data
            x_train_win = create_dataset(x_train_std[:, i].reshape(-1, 1), look_back)
            # valid data
            x_valid_win = create_dataset(x_valid_std[:, i].reshape(-1, 1), look_back)
            # test data
            x_test_win = create_dataset(x_test_std[:, i].reshape(-1, 1), look_back)
        else:
            # train data
            x_train_win = np.concatenate([x_train_win, create_dataset( \
                x_train_std[:, i].reshape(-1, 1), look_back)], axis=-1)
            # valid data
            x_valid_win = np.concatenate([x_valid_win, create_dataset( \
                x_valid_std[:, i].reshape(-1, 1), look_back)], axis=-1)
            # test data
            x_test_win = np.concatenate([x_test_win, create_dataset( \
                x_test_std[:, i].reshape(-1, 1), look_back)], axis=-1)

    # change the shape of data
    train_x_win = x_train_win.reshape(-1, 3 * data_dim)
    train_y = y_train[look_back - 1:]

    valid_x_win = x_valid_win.reshape(-1, 3 * data_dim)
    valid_y = y_valid[look_back - 1:]

    test_x_win = x_test_win.reshape(-1, 3 * data_dim)
    test_y = y_test[look_back - 1:]

    # change data type of _x_win from ndarray into dataframe to calculate the importance of characteristic.
    features = ['A1_mean', 'A1_min', 'A1_max', \
                'A2_mean', 'A2_min', 'A2_max', \
                'Cur_mean', 'Cur_min', 'Cur_max', \
                'Pre_mean', 'Pre_min', 'Pre_max', \
                'Temp_mean', 'Temp_min', 'Temp_max', \
                'Ther_mean', 'Ther_min', 'Ther_max', \
                'Vol_mean', 'Vol_min', 'Vol_max', \
                'Flow_mean', 'Flow_min', 'Flow_max']

    train_x = pd.DataFrame(train_x_win, columns=features)
    valid_x = pd.DataFrame(valid_x_win, columns=features)
    test_x = pd.DataFrame(test_x_win, columns=features)
    print(train_x)
    results_list = []
    if hypo == "random":
        # Ramdom search for hyper parameter
        optimization_trial = 100

        results_val_acc = {}
        results_train_acc = {}

        for _ in range(optimization_trial):
            # =====the searching area of hyper parameter =====
            lr = 10 ** np.random.uniform(min_lr, max_lr)
            min_data_in_leaf = np.random.choice(range(min_min_data_in_leaf, max_min_data_in_leaf), 1)[0]
            max_depth = np.random.choice(range(min_max_depth, max_max_depth), 1)[0]
            num_leaves = np.random.choice(range(min_num_leaf, max_num_leaf), 1)[0]
            # ================================================

            # Hyper parameter
            lgb_params = {'objective': object,
                  'metric': metric,
                  'force_row_wise': True,
                  'seed': seed,
                  'learning_rate': lr,
                  'min_data_in_leaf': min_data_in_leaf,
                  'max_depth': max_depth,
                  'num_leaves': num_leaves
                  }

            train_acc, valid_acc = lgb_train_predict(train_x, train_y, valid_x, valid_y, test_x, test_y, params=lgb_params,
                                             test_flag=False)
            print('optimization' + str(len(results_val_acc) + 1))
            print("train acc:" + str(train_acc) + "valid acc:" + str(valid_acc) + " | lr:" + str(
            lr) + ", min_data_in_leaf:" + str(min_data_in_leaf) + \
              ",max_depth:" + str(max_depth) + ",num_leaves:" + str(num_leaves))

            key = " lr:" + str(lr) + ", min_data_in_leaf:" + str(min_data_in_leaf) + ", max_depth:" + str(
            max_depth) + ",num_leaves:" + str(num_leaves)
            results_val_acc[key] = valid_acc
            results_train_acc[key] = train_acc
    elif hypo == "grid":
        print("a lot of combination to use grid search try to use random search")

    print("=========== Hyper-Parameter Optimization Result ===========")
    i = 0
    best_parm = []
    for key, val_acc in sorted(results_val_acc.items(), key=lambda x: x[1], reverse=True):

        print("Best-" + str(i + 1) + "(val acc:" + str(val_acc) + ")| " + key)
        if i ==0:
            best_parm.append(key)
        i += 1

        if i >= int(optimization_trial * 0.05):
            break

    str_split_best_parm = best_parm[0].split(",")
    print("best_parm:",best_parm)
    for i in range(4):
        final_split_best_parm = str_split_best_parm[i].split(":")
        results_list.append(float(final_split_best_parm[1]))
    print(final_split_best_parm)
    print(results_list)

    # fine-tunned hyper paramter
    lgb_params = {'objective': 'binary',
                  'metric': 'binary_error',
                  'force_row_wise': True,
                  'seed': 0,
                  'learning_rate': results_list[0],
                  'min_data_in_leaf': int(results_list[1]),
                  'max_depth': int(results_list[2]),
                  'num_leaves': int(results_list[3])
                  }

    test_acc, test_f1score, test_cm, test_pred, model_lgb = lgb_train_predict(train_x, train_y, valid_x, valid_y,
                                                                              test_x, test_y, params=lgb_params,
                                                                              test_flag=True)

    print('test_acc:' + str(test_acc))
    print('test_f1score:' + str(test_f1score))
    print('test_confusionMatrix:')
    print(test_cm)