import os
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import sys
import matplotlib.pyplot as plt
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
from io import StringIO



def analyze_data(full_tab_with_clust_embed, dest_path):
    overall_train_set_no_embed = full_tab_with_clust_embed.drop(["embeddings"], axis=1)
    print(overall_train_set_no_embed.groupby("representative").nunique("nsub").groupby("nsub").size(),
          "different nsubs in cluster")
    overall_train_set_no_embed.groupby("representative").nunique("nsub").groupby("nsub").size().plot(kind='bar',
                                                                                                     grid=False,
                                                                                                     log=True,
                                                                                                     color="maroon",
                                                                                                     fontsize=10,
                                                                                                     title="different oligomeric states per sequence similarity cluster",
                                                                                                     xlabel="different oligomeric states",
                                                                                                     ylabel="number of clusters")
    plt.savefig(dest_path + "different_oligomeric_states_per_cluster.png")



def probability_analysis(y_prob, y_test, dest_path):
    y_prob_df = pd.DataFrame(y_prob)
    inv_map = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 10, 9: 12, 10: 14, 11: 24}
    y_prob_df.rename(columns=inv_map, inplace=True)
    y_prob_df.apply(lambda x: x.nlargest(3).index, axis=1, result_type='expand')
    proba_pred_actual = pd.concat((y_prob_df.apply(lambda x: x.nlargest(3).index, axis=1, result_type='expand'), y_test.reset_index().nsub.astype(int)), axis=1)
    print(proba_pred_actual[0].eq(proba_pred_actual['nsub']).sum())
    print(proba_pred_actual[1].eq(proba_pred_actual['nsub']).sum())
    print(proba_pred_actual[2].eq(proba_pred_actual['nsub']).sum())
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
    inv_map = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 10, 9: 12, 10: 14, 11: 24}
    mat_df.rename(inv_map, axis=1, inplace=True)
    mat_df.rename(inv_map, inplace=True)
    mat_df_per = mat_df.div(mat_df.sum(axis=1), axis=0).round(2)
    s = sns.heatmap(mat_df_per, annot=True, fmt='g', cmap="BuPu", vmin=0, vmax=1)
    s.figure.set_size_inches(8, 6)
    s.set(xlabel='Prediction', ylabel='Actual_lables', title=title)
    plt.savefig(dest_path + title + ".png")
    # plt.show()
    plt.close()



if __name__ == "__main__":
    full_tab_with_clust_embed = pd.read_pickle(
        "/vol/ek/Home/orlyl02/working_dir/oligopred/clustering/re_clust_c0.3/full_tab_with_clust_embed.pkl")
    full_tab_with_clust_embed.reset_index(drop=True, inplace=True)
    dest_path = "/vol/ek/Home/orlyl02/working_dir/oligopred/mlp/cov03_runs/analyzing_models/analyze_model_mlp_clf_8020_downsample3_2022-12-21/"
    y_prob = pd.read_pickle("/vol/ek/Home/orlyl02/working_dir/oligopred/mlp/cov03_runs/proba_8020_downsample3.pkl")
    y_test = pd.read_pickle("cov03_runs/y_test2_20_downsample3.pkl")
    y_test.reset_index(drop=True, inplace=True)
    proba_pred_actual = probability_analysis(y_prob, y_test, dest_path)
    print(y_prob)

