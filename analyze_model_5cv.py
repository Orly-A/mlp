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

NUM_CV = 5


"""
This script is used to analyze the results of the xgboost model.
the first arg is the wanted model full path
then 2,3,4,5 are X_train, y_train, X_test, y_test
"""

def open_the_data_and_general_figs():

    # hardcoded model
    model_name = "mlp_clf_8020_downsample3_5cv"
    today = str(date.today())
    dest_path = "/vol/ek/Home/orlyl02/working_dir/oligopred/mlp/esm_runs/analyzing_models/analyze_model_" + model_name + "_" + today + "/"

    # check if the directory exists, if not create it
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)

    # new data from esm2
    # overall_train_set = pd.read_pickle("/vol/ek/Home/orlyl02/working_dir/oligopred/esmfold_prediction/train_set_c0.3.pkl")
    # used since we downsample and the overall set is different
    overall_train_set = pd.read_pickle("/vol/ek/Home/orlyl02/working_dir/oligopred/mlp/esm_runs/5cv_sets_proba_models/overall_train_set_downsampled3_5cv.pkl")


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
    # xgb_joblib, X_train, y_train, X_test, y_test = get_data_from_args()
    y_proba = load_data_from_5_cvs()
    # load the data from all the cvs

    return y_proba, dest_path, overall_train_set


def load_data_from_5_cvs():
    final_results = pd.DataFrame()
    for i in range(NUM_CV):
        # read the pickle data
        # with open("/vol/ek/Home/orlyl02/working_dir/oligopred/mlp/esm_runs/X_train%s_80_5cv.pkl" % i, "rb") as f:
        #     X_train = pickle.load(f)
        #     overall_X_train.append(X_train)
        # with open("/vol/ek/Home/orlyl02/working_dir/oligopred/mlp/esm_runs/y_train%s_80_5cv.pkl" % i, "rb") as f:
        #     y_train = pickle.load(f)
        #     overall_y_train = pd.concat([overall_y_train, y_train])
        # with open("/vol/ek/Home/orlyl02/working_dir/oligopred/mlp/esm_runs/X_test%s_20_5cv.pkl" % i, "rb") as f:
        #     X_test = pickle.load(f)
        #     overall_X_test.append(X_test)
        with open("/vol/ek/Home/orlyl02/working_dir/oligopred/mlp/esm_runs/5cv_sets_proba_models/y_test%s_20_5cv.pkl" % i, "rb") as f:
            y_test = pickle.load(f)
        with open("/vol/ek/Home/orlyl02/working_dir/oligopred/mlp/esm_runs/5cv_sets_proba_models/proba_8020_downsample3_cv%s.pkl" % i, "rb") as f:
            y_proba = pickle.load(f)
        y_proba_df = pd.DataFrame(y_proba)
        with open("/vol/ek/Home/orlyl02/working_dir/oligopred/mlp/esm_runs/5cv_sets_proba_models/y_labels%s_20_5cv.pkl" % i, "rb") as f:
            y_labels = pickle.load(f)
        merged = pd.concat([y_test.reset_index(), y_proba_df], axis=1)
        merged_with_labels = pd.concat([merged, y_labels.reset_index()], axis=1)
        final_results = pd.concat([final_results, merged_with_labels])
    return final_results


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
    class_report_df.index = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 24, "accuracy", "macro_avg", "weighted_avg"]
    pickle.dump(class_report_df, open(dest_path+"class_report_df.pkl", "wb"))

# plot the confusion matrix using the probabilities
def top2_analysis(overall_train_set, y_prob, dest_path, inv_map):
    # currently seems ok!! :)
    y_prob.rename(columns=inv_map, inplace=True)
    y_prob.reset_index(inplace=True, drop=True)
    overall_train_set.reset_index(inplace=True, drop=True)
    y_prob_with_overall_train = y_prob.merge(overall_train_set, how="left", on=["code", "nsub"])
    y_prob_with_overall_train = y_prob_with_overall_train[['nsub', 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 24, 'code', 'representative']]
    convert_dict = {"nsub": float, 1: float, 2: float, 3: float, 4: float, 5: float, 6: float, 7: float, 8: float,
                    10: float, 12: float, 14: float, 24: float}
    y_prob_with_overall_train = y_prob_with_overall_train.astype(convert_dict)
    pickle.dump(y_prob_with_overall_train, open("/vol/ek/Home/orlyl02/working_dir/oligopred/mlp/esm_runs/5cv_sets_proba_models/y_prob_with_overall_train.pkl", "wb"))

    proba_pred_actual = pd.concat((y_prob_with_overall_train[[1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 24]]
                                   .apply(lambda x: x.nlargest(3).index, axis=1, result_type='expand'),
                                   y_prob_with_overall_train.nsub.astype(int)), axis=1)

    proba_pred_actual['top_2'] = np.where(proba_pred_actual[1] == proba_pred_actual["nsub"], proba_pred_actual[1], proba_pred_actual[0])
    top2_bal_acc = round(metrics.balanced_accuracy_score(proba_pred_actual['nsub'].astype(int), proba_pred_actual['top_2'].astype(int), adjusted=True), 3)
    gen_con_mat_and_fig(metrics.confusion_matrix(proba_pred_actual['nsub'].astype(int), proba_pred_actual['top_2'].astype(int)), dest_path,
                     ("top_2_confusion_matrix Adjusted_balanced_accuracy: " + str(top2_bal_acc)), inv_map)
    regular_bal_acc = round(metrics.balanced_accuracy_score(proba_pred_actual['nsub'].astype(int), proba_pred_actual[0].astype(int), adjusted=True), 3)
    gen_con_mat_and_fig(metrics.confusion_matrix(proba_pred_actual['nsub'].astype(int), proba_pred_actual[0].astype(int)), dest_path,
                       ("regular_confusion_matrix Adjusted_balanced_accuracy: " + str(regular_bal_acc)), inv_map)

    return proba_pred_actual

def gen_con_mat_and_fig(mat, dest_path, title, inv_map):
    mat_df = pd.DataFrame(mat)
    #check this is the correct mapping
    # inv_map = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 8, 7: 10, 8: 12, 9: 24}
    mat_df.rename(inv_map, axis=1, inplace=True)
    mat_df.rename(inv_map, inplace=True)
    mat_df_per = mat_df.div(mat_df.sum(axis=1), axis=0).round(2)
    s = sns.heatmap(mat_df_per, annot=True, fmt='g', cmap="BuPu", vmin=0, vmax=1)
    s.figure.set_size_inches(8, 6)
    s.set(xlabel='Prediction', ylabel='Actual_lables', title=title)
    plt.savefig(dest_path + title.split(" ")[0] + ".png")
    # plt.show()
    plt.close()
    s.clear()



if __name__ == "__main__":
    y_prob, dest_path, overall_train_set = open_the_data_and_general_figs()
    inv_map = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 10, 9: 12, 10: 14, 11: 24}
    proba_pred_actual = top2_analysis(overall_train_set, y_prob, dest_path, inv_map)
    y_pred = proba_pred_actual[0]
    y_test = proba_pred_actual["nsub"]
    class_report(y_test, y_pred, dest_path)
    pickle.dump(proba_pred_actual, open("/vol/ek/Home/orlyl02/working_dir/oligopred/mlp/esm_runs/5cv_sets_proba_models/proba_pred_actual.pkl", "wb"))
    print("done")
