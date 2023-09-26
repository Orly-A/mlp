import re
import os
import pickle
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
from datetime import date
import joblib

import seaborn as sns
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import LabelEncoder



"""
This script is used to analyze the results of the xgboost model.
the first arg is the wanted model full path
then 2,3,4,5 are X_train, y_train, X_test, y_test
"""

def open_the_data_and_general_figs():

    arg_pars = sys.argv[1:]
    # get the model name from the args
    model_name = arg_pars[0].split("/")[-1].split(".")[0]
    today = str(date.today())
    dest_path = "/vol/ek/Home/orlyl02/working_dir/oligopred/mlp/esm_runs/analyzing_models/analyze_model_" + model_name + "_" + today + "/"

    # check if the directory exists, if not create it
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)

    # with open("/vol/ek/Home/orlyl02/working_dir/oligopred/mlp/train_set.pkl", 'rb') as f:
    #     overall_train_set = pickle.load(f)
    # new data from esm2
    overall_train_set = pd.read_pickle("/vol/ek/Home/orlyl02/working_dir/oligopred/esmfold_prediction/train_set_c0.3.pkl")

    # index reset is important for the stratified splitting and the saving to lists
    overall_train_set.reset_index(drop=True, inplace=True)

    overall_train_set_no_embed = overall_train_set.drop(["esm_embeddings"], axis=1)
    a = overall_train_set_no_embed.groupby("representative").nunique("nsub").groupby("nsub").size().plot\
        (kind='bar', grid=False, log=True, color="maroon", fontsize=10,
         title="different oligomeric states per sequence similarity cluster", xlabel="different oligomeric states", ylabel="number of clusters")
    a.figure.savefig(dest_path + "different_oligomeric_states_per_cluster.png")
    b = overall_train_set_no_embed.groupby("representative").nunique("code").groupby("code").size().plot\
        (kind='bar', color="maroon", figsize=[20,7], fontsize=10, log=True, grid=False,
         title="number of different pdbs in each sequence similarity cluster", xlabel="number of unique protein sequences", ylabel="number of clusters")
    b.figure.savefig(dest_path + "number_of_unique_pdbs_per_cluster.png")
    b.clear()

    xgb_joblib = joblib.load(arg_pars[0])
    path_2_X_train = arg_pars[1]
    path_2_y_train = arg_pars[2]
    path_2_X_test = arg_pars[3]
    path_2_y_test = arg_pars[4]
    X_train = pickle.load(open(path_2_X_train, "rb"))
    y_train = pickle.load(open(path_2_y_train, "rb"))
    X_test = pickle.load(open(path_2_X_test, "rb"))
    y_test = pickle.load(open(path_2_y_test, "rb"))
    return xgb_joblib, X_train, y_train, X_test, y_test, dest_path


def get_model_results(xgb_joblib, X_train, y_train, X_test, y_test, dest_path):
    y_train = y_train.values.astype(int)
    y_test_for_decode = y_test.values.astype(int)
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    # y_test_transformed = le.fit_transform(y_test_for_decode)
    inv_map = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 10, 9: 12, 10: 14, 11: 24}
    transformed_y_train = np.array([inv_map[x] for x in y_train])
    #this map is used since there are no predicted 7 or 14 mers
    map = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 7, 10: 8, 12: 9, 24: 11}
    y_test_transformed = np.array([map[x] for x in y_test])


    y_pred = xgb_joblib.predict(X_test)
    y_prob = xgb_joblib.predict_proba(X_test)
    return y_pred, y_prob, y_test_transformed, transformed_y_train, y_test_for_decode, dest_path, inv_map, map

def results_numbers(y_test_transformed, y_pred, dest_path):

    result_dict = {}
    result_dict["adjusted Balanced accuracy"] = round(metrics.balanced_accuracy_score(y_test_transformed, y_pred, adjusted=True), 3)
    result_dict["Balanced accuracy"] = round(metrics.balanced_accuracy_score(y_test_transformed, y_pred, adjusted=False), 3)
    result_dict["f1_score"] = round(f1_score(y_test_transformed, y_pred, average='weighted'), 3)
    result_dict["precision"] = round(precision_score(y_test_transformed, y_pred, average='weighted'), 3)
    result_dict["recall"] = round(recall_score(y_test_transformed, y_pred, average='weighted'), 3)

    #save dict to csv
    with open(dest_path + "score_results.csv", 'w') as f:
        for key in result_dict:
            f.write(key + "," + str(result_dict[key]) + "\n")



def confs_matrix(y_test_transformed, y_pred, dest_path):
    nsub_labels = [0,1,2,3,4,5,6,7,8,9,10,11]

    conf_mat = metrics.confusion_matrix(y_test_transformed, y_pred, labels=nsub_labels)
    conf_mat_df = pd.DataFrame(conf_mat)
    # res = {}
    # for cl in le.classes_:
    #     res.update({cl:le.transform([cl])[0]})
    # inv_map = {v: k for k, v in res.items()}

    with open("/vol/ek/Home/orlyl02/working_dir/oligopred/mlp/esm_runs/le_dict2_cov03.pkl", 'rb') as f:
        le_dict = pickle.load(f)

    inv_map = {v: k for k, v in le_dict.items()}
    inv_map = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 10, 9: 12, 10: 14, 11: 24}
    #rename cols (predicted)
    conf_mat_df.rename(inv_map, axis=1, inplace=True)
    #rename rows (actual)
    conf_mat_df.rename(inv_map, inplace=True)
    pickle.dump(conf_mat_df, open(dest_path+"conf_mat_df.pkl", "wb"))
    conf_mat_df_percent = conf_mat_df.div(conf_mat_df.sum(axis=1), axis=0).round(2)
    s = sns.heatmap(conf_mat_df_percent, annot=True, fmt='g', cmap="BuPu", vmin=0, vmax=1)
    s.set(xlabel='Prediction', ylabel='Actual lables', title="Confusion matrix")
    # set the size of the plot
    s.figure.set_size_inches(8,6)
    s.figure.savefig(dest_path + "conf_mat_df_percent.png")
    s.clear()
    plt.close()
    pickle.dump(conf_mat_df.sum(axis=1).to_dict(), open(dest_path+"actual_counts_per_qs.pkl", "wb"))


def class_report(y_test_transformed, y_pred, dest_path):
    class_report = metrics.classification_report(y_test_transformed, y_pred, output_dict=True)
    class_report_df = pd.DataFrame(class_report).transpose()
    class_report_df = class_report_df.round(3)
    print(class_report_df)
    # class_report_df.index = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 24, "accuracy", "macro_avg", "weighted_avg"]
    class_report_df.index = [1, 2, 3, 4, 5, 6, 8, 10, 12, 24, "accuracy", "macro_avg", "weighted_avg"]
    pickle.dump(class_report_df, open(dest_path+"class_report_df.pkl", "wb"))

# plot the confusion matrix using the probabilities
def top2_analysis(y_prob, dest_path, inv_map):
    y_prob_df = pd.DataFrame(y_prob)
    inv_map = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 10, 9: 12, 10: 14, 11: 24}
    y_prob_df.rename(columns=inv_map, inplace=True)

    proba_pred_actual = pd.concat((y_prob_df.apply(lambda x: x.nlargest(3).index, axis=1, result_type='expand'),
                                   y_test.reset_index().nsub.astype(int)), axis=1)
    proba_pred_actual['top_2'] = np.where(proba_pred_actual[1] == proba_pred_actual["nsub"], proba_pred_actual[1], proba_pred_actual[0])
    top2_bal_acc = round(metrics.balanced_accuracy_score(proba_pred_actual['nsub'].astype(int), proba_pred_actual['top_2'].astype(int), adjusted=True), 3)
    gen_con_mat_and_fig(metrics.confusion_matrix(proba_pred_actual['nsub'].astype(int), proba_pred_actual['top_2'].astype(int)), dest_path,
                     ("top_2_confusion_matrix Adjusted_balanced_accuracy: " + str(top2_bal_acc)))
    regular_bal_acc = round(metrics.balanced_accuracy_score(proba_pred_actual['nsub'].astype(int), proba_pred_actual[0].astype(int), adjusted=True), 3)
    gen_con_mat_and_fig(metrics.confusion_matrix(proba_pred_actual['nsub'].astype(int), proba_pred_actual[0].astype(int)), dest_path,
                       ("regular_confusion_matrix Adjusted_balanced_accuracy: " + str(regular_bal_acc)))

    return proba_pred_actual

def gen_con_mat_and_fig(mat, dest_path, title):
    mat_df = pd.DataFrame(mat)
    #check this is the correct mapping
    inv_map = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 8, 7: 10, 8: 12, 9: 24}
    mat_df.rename(inv_map, axis=1, inplace=True)
    mat_df.rename(inv_map, inplace=True)
    mat_df_per = mat_df.div(mat_df.sum(axis=1), axis=0).round(2)
    s = sns.heatmap(mat_df_per, annot=True, fmt='g', cmap="BuPu", vmin=0, vmax=1)
    s.figure.set_size_inches(8, 6)
    s.set(xlabel='Prediction', ylabel='Actual_lables', title=title)
    plt.savefig(dest_path + title + ".png")
    # plt.show()
    plt.close()
    s.clear()



if __name__ == "__main__":
    mlp_cls, X_train, y_train, X_test, y_test, dest_path = open_the_data_and_general_figs()
    y_pred, y_prob, y_test_transformed, transformed_y_train, y_test_for_decode, dest_path, inv_map, map = get_model_results(mlp_cls, X_train, y_train, X_test, y_test, dest_path)
    results_numbers(y_test_transformed, y_pred, dest_path)
    confs_matrix(y_test_transformed, y_pred, dest_path)
    class_report(y_test_transformed, y_pred, dest_path)
    proba_pred_actual = top2_analysis(y_prob, dest_path, inv_map)