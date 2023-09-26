import re
import os
import pickle
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import datetime
import glob
import joblib

from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler


# HEAD_OF_DATA = 1000
# LR = 0.1
NUM_CV = 5
# NUM_ITER = 20
UNDERSAMPLE_FACTOR = 3



def data_definition():

    overall_train_set = pd.read_pickle("/vol/ek/Home/orlyl02/working_dir/oligopred/esmfold_prediction/train_set_c0.3.pkl")
    holdout_set = pd.read_pickle("/vol/ek/Home/orlyl02/working_dir/oligopred/esmfold_prediction/hold_out_set_c0.3.pkl")

    # index reset is important for the stratified splitting and the saving to lists
    overall_train_set.reset_index(drop=True, inplace=True)
    overall_train_set, remove_list = remove_small_groups(overall_train_set)
    overall_train_set = downsample_mjorities(overall_train_set)
    overall_train_set.reset_index(drop=True, inplace=True)
    pickle.dump(overall_train_set, open("/vol/ek/Home/orlyl02/working_dir/oligopred/mlp/holdout_analysis/overall_train_c03_final.pkl", "wb"))
    holdout_set.reset_index(drop=True, inplace=True)
    holdout_set = holdout_set[~holdout_set["nsub"].isin(remove_list)]
    print(holdout_set.nsub.unique())
    holdout_set.reset_index(drop=True, inplace=True)
    pickle.dump(holdout_set, open("/vol/ek/Home/orlyl02/working_dir/oligopred/mlp/holdout_analysis/hold_out_set_c03_final.pkl", "wb"))

    X_train = overall_train_set['esm_embeddings'].tolist()
    y_train = overall_train_set['nsub']
    X_test = holdout_set['esm_embeddings'].tolist()
    y_test = holdout_set['nsub']

    return X_train, y_train, X_test, y_test, remove_list


def downsample_mjorities(overall_train_set):
    new_1_count = int(overall_train_set[overall_train_set["nsub"] == 1].shape[0]/UNDERSAMPLE_FACTOR)
    new_2_count = int(overall_train_set[overall_train_set["nsub"] == 2].shape[0]/UNDERSAMPLE_FACTOR)
    under_sample_dict = {1: new_1_count, 2: new_2_count}
    list_of_nsubs = list(set(overall_train_set["nsub"].tolist()))
    list_of_nsubs.remove(1)
    list_of_nsubs.remove(2)
    for nsub in list_of_nsubs:
        counter = int(overall_train_set[overall_train_set["nsub"] == nsub].shape[0])
        under_sample_dict[nsub] = counter
    print(under_sample_dict)
    rus = RandomUnderSampler(random_state=1, sampling_strategy=under_sample_dict)
    X, y = rus.fit_resample(overall_train_set[["code"]], overall_train_set["nsub"])
    overall_train_set = overall_train_set[overall_train_set.code.isin(X["code"].tolist())]
    return overall_train_set

def remove_small_groups(overall_train_set):
    overall_train_set_no_embed = overall_train_set[["code", "nsub", "representative"]]
    overall_train_set2 = overall_train_set.copy()
    list_of_nsubs = list(set(overall_train_set2["nsub"].tolist()))
    remove_list = []
    for nsub in list_of_nsubs:
        num_of_clusts = overall_train_set_no_embed[overall_train_set_no_embed['nsub'] == nsub].groupby("representative").nunique().shape[0]
        if num_of_clusts < NUM_CV:
            print(nsub, "nsub")
            print(num_of_clusts, "num_of_clusts")
            overall_train_set2 = overall_train_set2[overall_train_set2.nsub != nsub]
            remove_list.append(nsub)
    return overall_train_set2, remove_list


def remove_classes_hold_out(hold_out, remove_list):
    pass


def build_mlp_model(X_train, y_train, X_test, y_test):
    # params from the esm runs with downsampling=3
    clf = MLPClassifier(activation='identity', learning_rate='adaptive',
                        learning_rate_init=0.01, solver='adam',
                        max_iter=1000, n_iter_no_change=20,
                        tol=0.001, hidden_layer_sizes=(120,),
                        alpha=0.0005, batch_size=250,
                        random_state=22)

    y_train = y_train.values.astype(int)
    y_test = y_test.values.astype(int)
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.fit_transform(y_test)

    le_dict = dict(zip(le.classes_, le.transform(le.classes_)))
    print(le_dict)

    with open('/vol/ek/Home/orlyl02/working_dir/oligopred/mlp/holdout_analysis/le_dict2_cov03.pkl', 'wb') as f:
        pickle.dump(le_dict, f)


    print("starting the tuning")
    print(datetime.datetime.now())
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # save the final model
    pickle.dump(clf, open("/vol/ek/Home/orlyl02/working_dir/oligopred/mlp/holdout_analysis/mlp_clf_8020_downsample3_final.pkl", "wb"))
    print("finished the tuning")
    print(datetime.datetime.now())

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    result_dict = {}
    result_dict["Balanced_accuracy"] = round(metrics.balanced_accuracy_score(y_test, y_pred), 3)
    result_dict["f1_score"] = round(f1_score(y_test, y_pred, average='weighted'), 3)
    result_dict["precision"] = round(precision_score(y_test, y_pred, average='weighted'), 3)
    result_dict["recall"] = round(recall_score(y_test, y_pred, average='weighted'), 3)
    result_dict["RMSE"] = round(rmse, 3)

    print(result_dict)
    #save dict to csv
    with open("/vol/ek/Home/orlyl02/working_dir/oligopred/mlp/holdout_analysis/score_results_downsample3_final.csv", 'a') as f:
        for key in result_dict:
            f.write(key + "," + str(result_dict[key]) + "\n")

    proba = clf.predict_proba(X_test)
    # pickle.dump(proba, open("/vol/ek/Home/orlyl02/working_dir/oligopred/mlp/esm_runs/proba_8020_downsample3.pkl", "wb"))
    # save the proba with the cv num
    pickle.dump(proba, open("/vol/ek/Home/orlyl02/working_dir/oligopred/mlp/holdout_analysis/proba_8020_downsample3_final.pkl", "wb"))
    # print(proba)



if __name__ == "__main__":
    X_train, y_train, X_test, y_test, remove_list = data_definition()
    # build the model - regular run
    build_mlp_model(X_train, y_train, X_test, y_test)
    print(datetime.datetime.now())
