import re
import os
import pickle
import numpy as np
import pandas as pd
import sys
import joblib
# import matplotlib.pyplot as plt
# import datetime

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import RandomizedSearchCV
# from sklearn.metrics import mean_squared_error
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import precision_score
# from sklearn.metrics import recall_score
# from sklearn.dummy import DummyClassifier

import xgboost as xgb
from xgboost import XGBClassifier

from io import StringIO

pd.set_option('display.max_columns', 20)
NUM_CV = 5


def open_scores(file_name, hyper_param_list):
    """
    Open the scores file and return a dataframe.
    split the lines to the correct columns.
    calc the average of the scores for each model
    remove models with less than 5 rows
    """
    cols = ["activation", 'alpha', 'batch_size', 'hidden_layer_sizes', 'learning_rate', 'learning_rate_init',
            'max_iter', 'n_iter_no_change', 'solver', 'tol',
            'adjusted_bal_acc_test', "f1_score_test"]
    # use grep to get the lines that begin with [
    sys.command = "grep -e '^\[' " + file_name
    run_text = os.popen(sys.command).read()
    # print(run_text)
    # print(type(run_text))
    if run_text == '':
        return
    run_text = StringIO(run_text)

    # runlog = pd.read_table(run_text, sep=',', header=None, names=cols)
    runlog = pd.read_csv(run_text, sep=', ', header=None, names=cols)

    # make sure we have the right row count
    # assert runlog.shape[1] == 8
    # print(runlog)
    runlog[['cv', 'activation']] = runlog['activation'].str.split('=', 1, expand=True)
    runlog['alpha'] = runlog['alpha'].str.split('=', 1, expand=True)[1]
    runlog['alpha'] = runlog['alpha'].astype(float)
    runlog['batch_size'] = runlog['batch_size'].str.split('=', 1, expand=True)[1]
    runlog['batch_size'] = runlog['batch_size'].astype(int)
    runlog['hidden_layer_sizes'] = runlog['hidden_layer_sizes'].str.split('=', 1, expand=True)[1]#.str.replace(r'\D+', '', regex=True)
    # runlog['hidden_layer_sizes'] = runlog['hidden_layer_sizes'].astype(int)
    runlog['learning_rate'] = runlog['learning_rate'].str.split('=', 1, expand=True)[1]
    runlog['learning_rate_init'] = runlog['learning_rate_init'].str.split('=', 1, expand=True)[1]
    runlog['learning_rate_init'] = runlog['learning_rate_init'].astype(float)
    runlog['max_iter'] = runlog['max_iter'].str.split('=', 1, expand=True)[1]
    runlog['max_iter'] = runlog['max_iter'].astype(int)
    runlog['n_iter_no_change'] = runlog['n_iter_no_change'].str.split('=', 1, expand=True)[1]
    runlog['n_iter_no_change'] = runlog['n_iter_no_change'].astype(int)
    runlog['solver'] = runlog['solver'].str.split('=', 1, expand=True)[1]

    runlog[['tol', 'adjusted_bal_acc_train']] = runlog['tol'].str.split(';', 1, expand=True)
    runlog['tol'] = runlog['tol'].str.split('=', 1, expand=True)[1]
    runlog['tol'] = runlog['tol'].astype(float)



    runlog['adjusted_bal_acc_train'] = runlog['adjusted_bal_acc_train'].str.split('=', -1, expand=True)[1]
    runlog['adjusted_bal_acc_train'] = runlog['adjusted_bal_acc_train'].astype(float)
    # split the score_test string into two parts, each containing a number
    runlog[['adjusted_bal_acc_test', 'f1_score_train']] = runlog['adjusted_bal_acc_test'].str.split(')', 1, expand=True)
    # for score_test leave only the number on the right side of the equal sign
    runlog['adjusted_bal_acc_test'] = runlog['adjusted_bal_acc_test'].str.split('=', 1, expand=True)[1]
    runlog['adjusted_bal_acc_test'] = runlog['adjusted_bal_acc_test'].astype(float)
    runlog['f1_score_train'] = runlog['f1_score_train'].str.split('=', 1, expand=True)[1]
    runlog['f1_score_train'] = runlog['f1_score_train'].astype(float)
    runlog[['f1_score_test', 'run_time']] = runlog['f1_score_test'].str.split(')', 1, expand=True)
    runlog['f1_score_test'] = runlog['f1_score_test'].str.split('=', 1, expand=True)[1]
    runlog['f1_score_test'] = runlog['f1_score_test'].astype(float)
    # for run_time leave only the number on the right side of the equal sign
    runlog['run_time'] = runlog['run_time'].str.split('=', 1, expand=True)[1]
    runlog['run_time'] = runlog['run_time'].str.replace(r'd+\.\d+', '', regex=True)
    runlog['run_time'] = runlog['run_time'].str.split('.', 1, expand=True)[0]
    runlog['run_time'] = runlog['run_time'].astype(int)
    # for cv split on tge closed square bracket
    runlog['cv'] = runlog['cv'].str.split(']', 1, expand=True)[0]
    runlog['cv'] = runlog['cv'].str.split(' ', 1, expand=True)[1]

    runlog["mean_bal_acc_train"] = runlog.groupby(hyper_param_list)["adjusted_bal_acc_train"].transform("mean")
    runlog["mean_bal_acc_test"] = runlog.groupby(hyper_param_list)["adjusted_bal_acc_test"].transform("mean")
    runlog["mean_f1_score_train"] = runlog.groupby(hyper_param_list)["f1_score_train"].transform("mean")
    runlog["mean_f1_score_test"] = runlog.groupby(hyper_param_list)["f1_score_test"].transform("mean")

    runlog["mean_run_time"] = runlog.groupby(hyper_param_list)["run_time"].transform("mean")
    # Check how many rows for each model and remove the ones with less than 5 rows
    runlog["count_bal_acc_test"] = runlog.groupby(hyper_param_list)["adjusted_bal_acc_test"].transform("count")
    runlog["count_f1_score_test"] = runlog.groupby(hyper_param_list)["f1_score_test"].transform("count")

    runlog.drop(runlog[runlog["count_bal_acc_test"] < 5].index, inplace=True)
    # drop duplicate rows based on hyper_param_list
    runlog.drop_duplicates(hyper_param_list, inplace=True)
    return runlog


def analyze_scores_choose_params(scores_file, hyper_param_list):
    """
    open the aggregated score file
    check how many different models there are
    calculate some stats and distributions
    find the best model params
    :return: dict (?) with params to use
    """
    score_tab1 = scores_file.groupby(hyper_param_list)["mean_bal_acc_test"].mean().reset_index(). \
        sort_values(by=["mean_bal_acc_test"], ascending=False)
    print(score_tab1.head(50))
    score_tab2 = scores_file.groupby(hyper_param_list)["mean_f1_score_test"].mean().reset_index(). \
        sort_values(by=["mean_f1_score_test"], ascending=False)
    print(score_tab2.head(50))
    score_tab = pd.merge(score_tab1, score_tab2, on=hyper_param_list, how="inner")
    score_tab.sort_values(by=["mean_bal_acc_test"], ascending=False, inplace=True)
    best_params = score_tab.iloc[0][:-1].to_dict()
    print(best_params)
    return best_params, score_tab


def define_data():
    with open("/vol/ek/Home/orlyl02/working_dir/oligopred/mlp/train_set.pkl", 'rb') as f:
        overall_train_set = pickle.load(f)
    # index reset is important for the stratified splitting and the saving to lists
    overall_train_set.reset_index(drop=True, inplace=True)
    overall_train_set = remove_small_groups(overall_train_set)
    X = pd.DataFrame(np.vstack(overall_train_set['embeddings']))
    y = overall_train_set["nsub"]
    groups = overall_train_set["representative"]
    cv = StratifiedGroupKFold(n_splits=NUM_CV)
    df = pd.DataFrame(np.vstack(X))
    # convert_dict = gen_converter()
    # y = y.map(convert_dict)
    return X, y, groups, cv, df


def remove_small_groups(overall_train_set):
    overall_train_set_no_embed = overall_train_set[["code", "nsub", "representative"]]
    overall_train_set2 = overall_train_set.copy()
    list_of_nsubs = list(set(overall_train_set2["nsub"].tolist()))
    for nsub in list_of_nsubs:
        num_of_clusts = overall_train_set_no_embed[overall_train_set_no_embed['nsub'] == nsub].groupby(
            "representative").nunique().shape[0]
        if num_of_clusts < NUM_CV:
            # print(nsub, "nsub")
            # print(num_of_clusts, "num_of_clusts")
            overall_train_set2 = overall_train_set2[overall_train_set2.nsub != nsub]
    return overall_train_set2


def train_from_hyp(best_params, X, y, groups):
    xgb_model = XGBClassifier(objective='multi:softprob')

    y = y.values.astype(int)
    le = LabelEncoder()
    y = le.fit_transform(y)

    xgb_model.set_params(**best_params)

    # generate a scorer suitable for this kind of data
    f1_score_weighted = make_scorer(f1_score, average="weighted")
    f1_score_weighted

    xgb_model.fit(X, y)

    # Save the final model
    pickle.dump(xgb_model, open("/vol/ek/Home/orlyl02/working_dir/oligopred/mlp/mlp_model.pkl", "wb"))
    joblib.dump(xgb_model, "/vol/ek/Home/orlyl02/working_dir/oligopred/mlp/mlp_model.joblib")


if __name__ == "__main__":
    # get the path to the working directory
    working_dir = "/vol/ek/Home/orlyl02/working_dir/oligopred/mlp/esm_runs"
    # working_dir = "/vol/ek/Home/orlyl02/working_dir/oligopred/mlp/cov03_runs/three_class_tune"
    log_files = [f for f in os.listdir(working_dir) if re.search(r'picasso_esm_downsample3.*.log', f) and f.endswith('.log')]
    # the next line just gets one file, for checking purposes
    # log_files = [f for f in os.listdir(working_dir) if re.search(r'k5a.log', f) and f.endswith('.log')]
    hyper_param_list = ["activation", 'learning_rate', 'learning_rate_init', 'solver', 'max_iter',
                        'n_iter_no_change', 'tol', 'hidden_layer_sizes', 'alpha', 'batch_size']
    # create a dataframe for the scores
    scores_file = pd.DataFrame()
    # loop over all the log files
    for log_file in log_files:
        print(log_file)
        # open the log file
        runlog = open_scores(os.path.join(working_dir, log_file), hyper_param_list)
        # append the runlog to the scores dataframe
        scores_file = scores_file.append(runlog)
    scores_file.reset_index(inplace=True)
    # save the scores dataframe to a pickle file
    scores_file.to_pickle(os.path.join(working_dir, "raw_scores_downsample3_cov03.pkl"))
    print(scores_file)
    print(scores_file.shape)
    best_params, score_tab = analyze_scores_choose_params(scores_file, hyper_param_list)
    score_tab.to_pickle(os.path.join(working_dir, "score_tab_downsample_cov03.pkl"))
    X, y, groups, cv, df = define_data()
    # train_from_hyp(best_params, X, y, groups)

