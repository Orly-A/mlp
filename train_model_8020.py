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
    #check if the files exist
    # if os.path.exists("/vol/ek/Home/orlyl02/working_dir/oligopred/mlp/cov03_runs/X_train2_80.pkl"):
    #     X_train = pickle.load(open("/vol/ek/Home/orlyl02/working_dir/oligopred/mlp/cov03_runs/X_train2_80.pkl", "rb"))
    #     x_test = pickle.load(open("/vol/ek/Home/orlyl02/working_dir/oligopred/mlp/cov03_runs/X_test2_20.pkl", "rb"))
    #     y_train = pickle.load(open("/vol/ek/Home/orlyl02/working_dir/oligopred/mlp/cov03_runs/y_train2_80.pkl", "rb"))
    #     y_test = pickle.load(open("/vol/ek/Home/orlyl02/working_dir/oligopred/mlp/cov03_runs/y_test2_20.pkl", "rb"))
    #     return X_train, y_train, x_test, y_test


    # overall_train_set = pd.read_pickle("/vol/ek/Home/orlyl02/working_dir/oligopred/clustering/re_clust_c0.3/train_set_c0.3.pkl")
    overall_train_set = pd.read_pickle("/vol/ek/Home/orlyl02/working_dir/oligopred/esmfold_prediction/train_set_c0.3.pkl")

    # index reset is important for the stratified splitting and the saving to lists
    overall_train_set.reset_index(drop=True, inplace=True)

    overall_train_set = remove_small_groups(overall_train_set)
    overall_train_set = downsample_mjorities(overall_train_set)
    # overall_train_set = calc_weights(overall_train_set)
    overall_train_set.reset_index(drop=True, inplace=True)
    pickle.dump(overall_train_set, open("/vol/ek/Home/orlyl02/working_dir/oligopred/mlp/esm_runs/overall_train_set_downsampled3_5cv.pkl", "wb"))
    X = overall_train_set["code"]
    y = overall_train_set["nsub"]
    groups = overall_train_set["representative"]
    # weights = overall_train_set["weights"]

    cv = StratifiedGroupKFold(n_splits=NUM_CV, shuffle=True, random_state=1)
    train_lst = []
    test_lst = []
    for train_idxs, test_idxs in cv.split(X, y, groups):
        train_lst.append(X[train_idxs].tolist())
        test_lst.append(X[test_idxs].tolist())

    train_idx_df = pd.DataFrame(train_lst).transpose()
    train_idx_df.rename(columns={0:"train_0", 1:"train_1", 2:"train_2", 3:"train_3", 4:"train_4"}, inplace=True)
    test_idx_df = pd.DataFrame(test_lst).transpose()
    test_idx_df.rename(columns={0:"test_0", 1:"test_1", 2:"test_2", 3:"test_3", 4:"test_4"}, inplace=True)

    merged_train_test = pd.concat([train_idx_df, test_idx_df], axis=1, join="outer")
    merged_train_test.reset_index(drop=True, inplace=True)
    for i in range(NUM_CV):
        gen_and_save_fold(overall_train_set, merged_train_test, i)

    # train_set = overall_train_set[overall_train_set["code"].isin(merged_train_test["train_2"])]
    # test_set = overall_train_set[overall_train_set["code"].isin(merged_train_test["test_2"])]
    # X_train = train_set['esm_embeddings'].tolist()
    # y_train = train_set['nsub']
    # # weights_train = train_set['weights']
    #
    # X_test = test_set['esm_embeddings'].tolist()
    # y_test = test_set['nsub']
    #
    # pickle.dump(X_train, open("/vol/ek/Home/orlyl02/working_dir/oligopred/mlp/esm_runs/X_train2_80_downsample3.pkl", "wb"))
    # pickle.dump(y_train, open("/vol/ek/Home/orlyl02/working_dir/oligopred/mlp/esm_runs/y_train2_80_downsample3.pkl", "wb"))
    # pickle.dump(X_test, open("/vol/ek/Home/orlyl02/working_dir/oligopred/mlp/esm_runs/X_test2_20_downsample3.pkl", "wb"))
    # pickle.dump(y_test, open("/vol/ek/Home/orlyl02/working_dir/oligopred/mlp/esm_runs/y_test2_20_downsample3.pkl", "wb"))
    # return X_train, y_train, X_test, y_test

def gen_and_save_fold(overall_train_set, merged_train_test, i):
    train_set = overall_train_set[overall_train_set["code"].isin(merged_train_test["train_%s" % i])]
    test_set = overall_train_set[overall_train_set["code"].isin(merged_train_test["test_%s" % i])]
    X_train = train_set['esm_embeddings'].tolist()
    y_train = train_set['nsub']
    # weights_train = train_set['weights']

    X_test = test_set['esm_embeddings'].tolist()
    y_test = test_set['nsub']
    y_labels = test_set['code']

    pickle.dump(X_train, open("/vol/ek/Home/orlyl02/working_dir/oligopred/mlp/esm_runs/X_train%s_80_5cv.pkl" % i, "wb"))
    pickle.dump(y_train, open("/vol/ek/Home/orlyl02/working_dir/oligopred/mlp/esm_runs/y_train%s_80_5cv.pkl" % i, "wb"))
    pickle.dump(X_test, open("/vol/ek/Home/orlyl02/working_dir/oligopred/mlp/esm_runs/X_test%s_20_5cv.pkl" % i, "wb"))
    pickle.dump(y_test, open("/vol/ek/Home/orlyl02/working_dir/oligopred/mlp/esm_runs/y_test%s_20_5cv.pkl" % i, "wb"))
    pickle.dump(y_labels, open("/vol/ek/Home/orlyl02/working_dir/oligopred/mlp/esm_runs/y_labels%s_20_5cv.pkl" % i, "wb"))


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
    for nsub in list_of_nsubs:
        num_of_clusts = overall_train_set_no_embed[overall_train_set_no_embed['nsub'] == nsub].groupby("representative").nunique().shape[0]
        if num_of_clusts < NUM_CV:
            print(nsub, "nsub")
            print(num_of_clusts, "num_of_clusts")
            overall_train_set2 = overall_train_set2[overall_train_set2.nsub != nsub]
    return overall_train_set2

def calc_weights(overall_train_set):
    nsub_dict = {}
    nsub_list = sorted(overall_train_set.nsub.unique().tolist())
    for nsub in nsub_list:
        # print(nsub, overall_train_set[overall_train_set["nsub"] == nsub].shape[0])
        nsub_dict[nsub]=overall_train_set[overall_train_set["nsub"] == nsub].shape[0]
    for k, v in nsub_dict.items():
        nsub_dict[k] = round(20/v, 4)
    overall_train_set["weights"] = np.sqrt(overall_train_set.nsub.map(nsub_dict))
#    overall_train_set["weights"] = overall_train_set.nsub.map(nsub_dict)

    # print(overall_train_set["weights"])
    # print(overall_train_set["weights"].unique())
    return overall_train_set


def build_mlp_model(X_train, y_train, X_test=None, y_test=None, i=None):
    # original model for the initial data devision
    # clf = MLPClassifier(activation="logistic", alpha=0.0001, batch_size=300, hidden_layer_sizes=(60,),
    #                     learning_rate="invscaling", max_iter=300, n_iter_no_change=40,
    #                     solver='adam', tol=0.0001, learning_rate_init=0.001, random_state=1)

    # new params from the cov03 runs
    # the current model from the cov03 runs use the params from the second best since this didn't reproduce the best results
    # random_state=22
    # clf = MLPClassifier(activation='logistic', learning_rate='adaptive',
    #                     learning_rate_init=0.01, solver='adam',
    #                     max_iter=1000, n_iter_no_change=40, tol=1.0,
    #                     hidden_layer_sizes=(10), alpha=0.0001,
    #                     batch_size=10, random_state=7)

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

    with open('le_dict2_cov03.pkl', 'wb') as f:
        pickle.dump(le_dict, f)


    print("starting the tuning")
    print(datetime.datetime.now())
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # pickle.dump(clf, open("/vol/ek/Home/orlyl02/working_dir/oligopred/mlp/esm_runs/mlp_clf_8020_downsample3.pkl", "wb"))
    # save the model with the cv num
    pickle.dump(clf, open("/vol/ek/Home/orlyl02/working_dir/oligopred/mlp/esm_runs/mlp_clf_8020_downsample3_cv{}.pkl".format(i), "wb"))

    # joblib.dump(clf, "/vol/ek/Home/orlyl02/working_dir/oligopred/mlp/esm_runs/mlp_clf_8020_downsample3.joblib")
    print("finished the tuning")
    print(datetime.datetime.now())

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    result_dict = {}
    result_dict["adjusted_Balanced_accuracy"] = round(metrics.balanced_accuracy_score(y_test, y_pred, adjusted=True), 3)
    result_dict["f1_score"] = round(f1_score(y_test, y_pred, average='weighted'), 3)
    result_dict["precision"] = round(precision_score(y_test, y_pred, average='weighted'), 3)
    result_dict["recall"] = round(recall_score(y_test, y_pred, average='weighted'), 3)
    result_dict["RMSE"] = round(rmse, 3)

    print(result_dict)
    #save dict to csv
    with open("/vol/ek/Home/orlyl02/working_dir/oligopred/mlp/esm_runs/score_results_downsample3_5cv.csv", 'a') as f:
        f.write("cv{}: ".format(i))
        for key in result_dict:
            f.write(key + "," + str(result_dict[key]) + "\n")

    proba = clf.predict_proba(X_test)
    # pickle.dump(proba, open("/vol/ek/Home/orlyl02/working_dir/oligopred/mlp/esm_runs/proba_8020_downsample3.pkl", "wb"))
    # save the proba with the cv num
    pickle.dump(proba, open("/vol/ek/Home/orlyl02/working_dir/oligopred/mlp/esm_runs/proba_8020_downsample3_cv{}.pkl".format(i), "wb"))
    # print(proba)


def build_mlp_model_with_umap(X_train, y_train, y_test, i):
    # import umap.umap_ as umap
    import umap
    import seaborn as sns
    from sklearn.preprocessing import StandardScaler
    from mpl_toolkits import mplot3d


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

    with open('le_dict2_cov03_umap3.pkl', 'wb') as f:
        pickle.dump(le_dict, f)


    # using umap dim reduction to reduce features in model
    s_reducer = umap.UMAP(verbose=True, n_components=3, n_neighbors=350,
                          min_dist=0.5)
    x = StandardScaler().fit_transform(X_train)
    s_umap_embeds = s_reducer.fit_transform(x, y_train)
    test_embedding = s_umap_embeds.transform(y_test)
    # s_umap_embeds_350_df = pd.DataFrame(data=s_umap_embeds)
    # finalDf_350 = pd.concat([s_umap_embeds_350_df, full_tab_with_clust_embed[['nsub', "code", 'representative']]], axis=1)
    # with open("/vol/ek/Home/orlyl02/working_dir/oligopred/dim_reduction/s_umap_embeds_350_full_data_esm.pkl", "wb") as f:
    #     pickle.dump(finalDf_350, f)
    # return finalDf_350
    X_train = s_umap_embeds.to_list()
    print("X_train", X_train)


    print("starting the tuning")
    print(datetime.datetime.now())
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # pickle.dump(clf, open("/vol/ek/Home/orlyl02/working_dir/oligopred/mlp/esm_runs/mlp_clf_8020_downsample3.pkl", "wb"))
    # save the model with the cv num
    pickle.dump(clf, open("/vol/ek/Home/orlyl02/working_dir/oligopred/mlp/esm_runs/5cv_sets_proba_models/mlp_clf_8020_downsample3_umap3_cv{}.pkl".format(i), "wb"))

    # joblib.dump(clf, "/vol/ek/Home/orlyl02/working_dir/oligopred/mlp/esm_runs/mlp_clf_8020_downsample3.joblib")
    print("finished the tuning")
    print(datetime.datetime.now())

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    result_dict = {}
    result_dict["adjusted_Balanced_accuracy"] = round(metrics.balanced_accuracy_score(y_test, y_pred, adjusted=True), 3)
    result_dict["f1_score"] = round(f1_score(y_test, y_pred, average='weighted'), 3)
    result_dict["precision"] = round(precision_score(y_test, y_pred, average='weighted'), 3)
    result_dict["recall"] = round(recall_score(y_test, y_pred, average='weighted'), 3)
    result_dict["RMSE"] = round(rmse, 3)

    print(result_dict)
    #save dict to csv
    with open("/vol/ek/Home/orlyl02/working_dir/oligopred/mlp/esm_runs/5cv_sets_proba_models/score_results_downsample3_umap3_5cv.csv", 'a') as f:
        f.write("cv{}: ".format(i))
        for key in result_dict:
            f.write(key + "," + str(result_dict[key]) + "\n")

    proba = clf.predict_proba(X_test)
    # pickle.dump(proba, open("/vol/ek/Home/orlyl02/working_dir/oligopred/mlp/esm_runs/proba_8020_downsample3.pkl", "wb"))
    # save the proba with the cv num
    pickle.dump(proba, open("/vol/ek/Home/orlyl02/working_dir/oligopred/mlp/esm_runs/5cv_sets_proba_models/proba_8020_downsample3_umap3_cv{}.pkl".format(i), "wb"))
    # print(proba)




if __name__ == "__main__":
    # X_train, y_train, X_test, y_test = data_definition()
    # data_definition()
    #for each cv run the model
    for i in range(NUM_CV):
        print("starting cv run: " + str(i))
        print(datetime.datetime.now())
        # read the pickle data
        with open("/vol/ek/Home/orlyl02/working_dir/oligopred/mlp/esm_runs/5cv_sets_proba_models/X_train%s_80_5cv.pkl" % i, "rb") as f:
            X_train = pickle.load(f)
        with open("/vol/ek/Home/orlyl02/working_dir/oligopred/mlp/esm_runs/5cv_sets_proba_models/y_train%s_80_5cv.pkl" % i, "rb") as f:
            y_train = pickle.load(f)
        with open("/vol/ek/Home/orlyl02/working_dir/oligopred/mlp/esm_runs/5cv_sets_proba_models/X_test%s_20_5cv.pkl" % i, "rb") as f:
            X_test = pickle.load(f)
        with open("/vol/ek/Home/orlyl02/working_dir/oligopred/mlp/esm_runs/5cv_sets_proba_models/y_test%s_20_5cv.pkl" % i, "rb") as f:
            y_test = pickle.load(f)

        # # build the model - regular run
        # build_mlp_model(X_train, y_train, X_test, y_test, i)
        # print("finished cv run: " + str(i))
        # print(datetime.datetime.now())
        # build the model - umap embeds
        build_mlp_model_with_umap(X_train, y_train, y_test, i)
        print("finished cv run: " + str(i))
        print(datetime.datetime.now())
    # build_mlp_model(X_train, y_train, X_test, y_test)
