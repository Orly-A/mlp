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


def prepare_data(proba_pred_actual, y_prob_with_overall_train, inv_map, PATH):
    multi_qs_clust = y_prob_with_overall_train.groupby("representative").nunique()[y_prob_with_overall_train.groupby("representative")
                                                                                       .nunique().nsub > 1].sort_values("nsub")
    relevant_clusters = multi_qs_clust[["nsub"]]
    proba_pred_actual.rename({0:'1_pred', 1:'2_pred', 2:'3_pred'}, axis=1, inplace=True)
    overall_proba_pred = pd.concat([y_prob_with_overall_train, proba_pred_actual[['1_pred', '2_pred', '3_pred', 'top_2']]], axis=1)
    multi_qs_tab = overall_proba_pred[overall_proba_pred["representative"].isin(relevant_clusters.index)]
    regular_bal_acc = round(metrics.balanced_accuracy_score(overall_proba_pred['nsub'].astype(int), overall_proba_pred["1_pred"].astype(int), adjusted=True), 3)
    gen_con_mat_and_fig(metrics.confusion_matrix(overall_proba_pred['nsub'].astype(int), overall_proba_pred["1_pred"].astype(int)), PATH,
                       ("regular_confusion_matrix Adjusted_balanced_accuracy: " + str(regular_bal_acc)), inv_map)
    multi_qs_bal_acc = round(metrics.balanced_accuracy_score(multi_qs_tab['nsub'].astype(int), multi_qs_tab["1_pred"].astype(int), adjusted=True), 3)
    gen_con_mat_and_fig(metrics.confusion_matrix(multi_qs_tab['nsub'].astype(int), multi_qs_tab["1_pred"].astype(int)), PATH,
                          ("multi_qs_confusion_matrix Adjusted_balanced_accuracy: " + str(multi_qs_bal_acc)), inv_map)
    single_qs_clust = y_prob_with_overall_train.groupby("representative").nunique()[y_prob_with_overall_train.groupby("representative")
                                                                                   .nunique().nsub == 1].sort_values("nsub")

    return multi_qs_clust, multi_qs_tab, single_qs_clust, overall_proba_pred, relevant_clusters



def cluster_analysis(multi_qs_clust, overall_proba_pred, PATH):
    cluster_summary_df = pd.DataFrame(columns=["representative", "num_of_qs", "nsub_classes", "num_of_pdbs", "adj_bal_ac", "adj_bal_ac_top2"])
    for rep in multi_qs_clust.index.to_list():
        tab = overall_proba_pred[overall_proba_pred["representative"] == rep]
        bal_ac = round(metrics.balanced_accuracy_score(tab['nsub'].astype(int), tab["1_pred"].astype(int), adjusted=True), 3)
        bal_ac_top2 = round(metrics.balanced_accuracy_score(tab['nsub'].astype(int), tab["top_2"].astype(int), adjusted=True), 3)
        cluster_summary_dict = {}
        cluster_summary_dict["representative"] = str(rep)
        cluster_summary_dict["num_of_qs"] = multi_qs_clust.loc[rep, "nsub"]
        cluster_summary_dict["nsub_classes"] = list(tab.nsub.unique())
        cluster_summary_dict["num_of_pdbs"] = tab.shape[0]
        cluster_summary_dict["adj_bal_ac"] = bal_ac
        cluster_summary_dict["adj_bal_ac_top2"] = bal_ac_top2
        cluster_summary_df = cluster_summary_df.append(cluster_summary_dict, ignore_index=True)
    cluster_summary_df.to_csv(PATH + "multi_qs_clusters_summary.csv", index=False, sep="\t")
    cluster_summary_df.to_pickle(PATH + "multi_qs_clusters_summary.pkl")
    return cluster_summary_df



def cluster_analysis_not_adjusted(qs_clust, overall_proba_pred, PATH, name):
    cluster_summary_df = pd.DataFrame(columns=["representative", "num_of_qs", "nsub_classes", "num_of_pdbs", "bal_ac", "bal_ac_top2"])
    for rep in qs_clust.index.to_list():
        tab = overall_proba_pred[overall_proba_pred["representative"] == rep]
        bal_ac = round(metrics.balanced_accuracy_score(tab['nsub'].astype(int), tab["1_pred"].astype(int)), 3)
        bal_ac_top2 = round(metrics.balanced_accuracy_score(tab['nsub'].astype(int), tab["top_2"].astype(int)), 3)
        cluster_summary_dict = {}
        cluster_summary_dict["representative"] = str(rep)
        cluster_summary_dict["num_of_qs"] = qs_clust.loc[rep, "nsub"]
        cluster_summary_dict["nsub_classes"] = list(tab.nsub.unique())
        cluster_summary_dict["num_of_pdbs"] = tab.shape[0]
        cluster_summary_dict["bal_ac"] = bal_ac
        cluster_summary_dict["bal_ac_top2"] = bal_ac_top2
        cluster_summary_df = cluster_summary_df.append(cluster_summary_dict, ignore_index=True)
    cluster_summary_df.to_csv(PATH + name + "_qs_clusters_summary_not_adjusted.csv", index=False, sep="\t")
    cluster_summary_df.to_pickle(PATH + name + "_qs_clusters_summary_not_adjusted.pkl")
    return cluster_summary_df



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

def num_of_qs_vs_predicted_num_of_qs(overall_proba_pred, PATH):
    num_qs_num_pred_qs = overall_proba_pred.groupby("representative").nunique().sort_values("nsub")[["nsub", "code", "1_pred"]]
    qs_dist_df = pd.DataFrame(columns=["1_pred", "nsub", "code"])
    for num in set(num_qs_num_pred_qs.nsub.to_list()):
        tab = num_qs_num_pred_qs[num_qs_num_pred_qs["nsub"] == num]
        a = tab.groupby("1_pred").count().reset_index()
        a["qs_in_cluster"] = num
        qs_dist_df = qs_dist_df.append(a)
    qs_dist_df.drop("code", axis=1, inplace=True)
    qs_dist_df.rename({"nsub": "num_pred_clusters"}, axis=1, inplace=True)
    qs_dist_pivot = qs_dist_df.pivot(index='qs_in_cluster', columns='1_pred', values='num_pred_clusters').fillna(0)
    qs_dist_pivot_percent = qs_dist_pivot.apply(lambda x: x / x.sum(), axis=1)
    qs_dist_pivot_percent.plot.bar(stacked=True, figsize=(9, 8), xlabel="different qs in cluster", ylabel="total clusters", title="num_of_qs_vs_predicted_num_of_qs", fontsize=10)
    legend = plt.legend(title="amount of predicted qs", loc='upper right', fontsize='small', fancybox=True)
    plt.savefig(PATH + "qs_distribution.png")

def bal_ac_vs_num_qs_per_clus(cluster_summary, name, PATH):
    pdbs_vs_ac = cluster_summary.groupby("num_of_pdbs").mean()["bal_ac"].reset_index()
    cluster_count = cluster_summary.groupby("num_of_pdbs").count().reset_index()[["num_of_pdbs", "representative"]]
    pdbs_vs_ac_with_count = cluster_count.merge(pdbs_vs_ac, on="num_of_pdbs")
    pdbs_vs_ac.plot(x="num_of_pdbs", y="bal_ac", kind="bar", figsize=(9, 8), title=(name + "qs in cluster balanced accuracy vs number of pdbs per cluster"))


def probabilities_analyses(y_prob_with_overall_train, PATH):
    # probability of each class by cluster and by nsub within
    # y_prob_with_overall_train.groupby(["representative", "nsub"]).mean()
    # biggest_clusters = y_prob_with_overall_train.groupby(["representative", "nsub"]).mean().groupby('representative').size().sort_values()[-3:-1].index.values
    # y_prob_with_overall_train.groupby(["representative", "nsub"]).mean().query("representative in @biggest_clusters").groupby(["representative", "nsub"]).plot.bar()
    #boxplots from the initial analysis with Matan
    for chosen_nsub in sorted(y_prob_with_overall_train.nsub.unique().tolist()):
        chosen_df = y_prob_with_overall_train[y_prob_with_overall_train.nsub == chosen_nsub][[1,2,3,4,5,6,7,8,10,12,14,24]]
        chosen_df.plot.box(title="'Confidence' in each label when actual label is " + str(chosen_nsub))
        plt.savefig(PATH + "confidence_in_each_label_when_actual_label_is_" + str(chosen_nsub) + ".png")
        plt.close()


def get_pisa_eppic_annot(overall_proba_pred, relevant_clusters, PATH):
    overall_qsbio = pd.read_csv("/vol/ek/Home/orlyl02/working_dir/oligopred/QSbio_PiQSi_annotations_V6_2020.csv", error_bad_lines=False, low_memory=False, skiprows=21)
    qsbio_relevant_rows = overall_qsbio[overall_qsbio["code"].isin(y_prob_with_overall_train.code.to_list())]
    qsbio_relevant_rows = qsbio_relevant_rows[["code", 'sym', 'PISA_identical', 'EPPIC_identical']]
    overall_set_proba_pisa = overall_proba_pred.merge(qsbio_relevant_rows, on="code", how="left")
    overall_set_proba_pisa["PISA_identical"] = pd.to_numeric(overall_set_proba_pisa["PISA_identical"], errors='coerce', downcast="integer")
    overall_set_proba_pisa["EPPIC_identical"] = pd.to_numeric(overall_set_proba_pisa["EPPIC_identical"], errors='coerce', downcast="integer")
    overall_set_proba_pisa["1_pred"] = pd.to_numeric(overall_set_proba_pisa["1_pred"], errors='coerce', downcast="integer")
    overall_set_proba_pisa["2_pred"] = pd.to_numeric(overall_set_proba_pisa["2_pred"], errors='coerce', downcast="integer")
    overall_set_proba_pisa["3_pred"] = pd.to_numeric(overall_set_proba_pisa["3_pred"], errors='coerce', downcast="integer")
    method_comp = overall_set_proba_pisa[["PISA_identical", "EPPIC_identical", "esm_identical", "nsub", "code", "representative"]]
    method_comp_grouped_nsub = method_comp.groupby("nsub").sum()
    method_comp_grouped_nsub_counts = method_comp_grouped_nsub.merge(method_comp.groupby("nsub").count()["representative"], on="nsub", how="left")
    method_comp_grouped_nsub_counts.rename({"representative": "count"}, axis=1, inplace=True)
    method_comp_grouped_nsub_counts_per = method_comp_grouped_nsub_counts.copy()
    method_comp_grouped_nsub_counts_per["PISA_identical"] = method_comp_grouped_nsub_counts["PISA_identical"]/method_comp_grouped_nsub_counts["count"]
    method_comp_grouped_nsub_counts_per["EPPIC_identical"] = method_comp_grouped_nsub_counts["EPPIC_identical"]/method_comp_grouped_nsub_counts["count"]
    method_comp_grouped_nsub_counts_per["esm_identical"] = method_comp_grouped_nsub_counts["esm_identical"]/method_comp_grouped_nsub_counts["count"]
    method_comp_grouped_nsub_counts_per[['PISA_identical', 'EPPIC_identical', 'esm_identical']].plot(kind="bar",
                                                                                                     figsize=(9, 8),
                                                                                                     title="success of each method by qs",
                                                                                                     xlabel="qs",
                                                                                                     ylabel="percentage of success",
                                                                                                     fontsize=10, ylim=([0,1]),
                                                                                                     cmap="viridis")

    plt.savefig(PATH + "success_of_each_method_by_qs.png")
    plt.close()
    multi_method_comp = method_comp[method_comp["representative"].isin(relevant_clusters.index)]
    multi_method_comp_grouped_nsub = multi_method_comp.groupby("nsub").sum()
    multi_method_comp_grouped_nsub_counts = multi_method_comp_grouped_nsub.merge(multi_method_comp.groupby("nsub").count()["representative"], on="nsub", how="left")
    multi_method_comp_grouped_nsub_counts.rename({"representative": "count"}, axis=1, inplace=True)
    multi_method_comp_grouped_nsub_counts_per = multi_method_comp_grouped_nsub_counts.copy()
    multi_method_comp_grouped_nsub_counts_per["PISA_identical"] = multi_method_comp_grouped_nsub_counts["PISA_identical"]/method_comp_grouped_nsub_counts["count"]
    multi_method_comp_grouped_nsub_counts_per["EPPIC_identical"] = multi_method_comp_grouped_nsub_counts["EPPIC_identical"]/method_comp_grouped_nsub_counts["count"]
    multi_method_comp_grouped_nsub_counts_per["esm_identical"] = multi_method_comp_grouped_nsub_counts["esm_identical"]/method_comp_grouped_nsub_counts["count"]
    multi_method_comp_grouped_nsub_counts_per[['PISA_identical', 'EPPIC_identical', 'esm_identical']].plot(kind="bar",
                                                                                                           figsize=(9, 8),
                                                                                                           title="success of each method by qs for multiclass clusters",
                                                                                                           xlabel="qs",
                                                                                                           ylabel="percentage of success",
                                                                                                           fontsize=10, ylim=([0,1]),
                                                                                                           cmap="viridis")
    plt.savefig(PATH + "success_of_each_method_by_qs_for_multiclass_clusters.png")
    plt.close()



def ecod(overall_proba_pred):
    ecod_20211004 = pd.read_csv("/vol/ek/share/databases/ecod/20211004/ecod.latest.domains.txt", sep='\t', skiprows=4)
    overall_proba_pred["pdb"] = overall_proba_pred["code"]
    overall_proba_pred["pdb"] = overall_proba_pred["pdb"].str.split("_", expand=True)[0]
    relevant_ecod = ecod_20211004[ecod_20211004["pdb"].isin(overall_proba_pred.pdb)]
    relevant_ecod = relevant_ecod.drop_duplicates(subset=["pdb", "f_name", "f_id"])
    relevant_ecod = relevant_ecod.drop("ligand", axis=1)
    relevant_ecod["length"] = relevant_ecod["seqid_range"]
    relevant_ecod["length"] = relevant_ecod["length"].str.split(":", expand=True)[1]
    relevant_ecod_length = pd.concat([relevant_ecod, relevant_ecod["length"].str.split("-", expand=True)], axis=1)
    relevant_ecod_length[0] = pd.to_numeric(relevant_ecod_length[0], errors='coerce', downcast="integer")
    relevant_ecod_length[1] = pd.to_numeric(relevant_ecod_length[1], errors='coerce', downcast="integer")
    relevant_ecod_length["seq_length"] =relevant_ecod_length[1] -relevant_ecod_length[0]
    relevant_ecod_length.drop(["length", 0, 1], axis=1, inplace=True)
    ###########
    #keeping only the ecod of the longest stretch for each pdb
    relevant_ecod_length = relevant_ecod_length.sort_values("seq_length").drop_duplicates("pdb", keep="first")
    overall_proba_pred_ecod = overall_proba_pred.merge(relevant_ecod_length, on="pdb", how="left")
    overall_proba_pred_ecod.to_csv(PATH + "overall_proba_pred_ecod.csv", sep="\t")
    clust_sizes = overall_proba_pred_ecod.groupby("f_name").nunique("nsub").groupby("nsub").size()
    print(clust_sizes)


def freq_by_label_in_multi_vs_all(overall_proba_pred_ecod):
    overall_dict = {}
    for chosen_nsub in sorted(overall_proba_pred_ecod.nsub.unique().tolist()):
        chosen_df = overall_proba_pred_ecod[overall_proba_pred_ecod.nsub == chosen_nsub][["1_pred", "2_pred", "3_pred"]]
        # a = chosen_df[(chosen_df["1_pred"] == chosen_nsub) | (chosen_df["2_pred"] == chosen_nsub) | (
        #             chosen_df["3_pred"] == chosen_nsub)]
        a = chosen_df[(chosen_df["1_pred"] == chosen_nsub) | (chosen_df["2_pred"] == chosen_nsub)]
        overall_dict[chosen_nsub] = a.shape[0]

    multi_dict = {}
    for chosen_nsub in sorted(multi_qs_tab.nsub.unique().tolist()):
        chosen_df = multi_qs_tab[multi_qs_tab.nsub == chosen_nsub][["1_pred", "2_pred", "3_pred"]]
        # a = chosen_df[(chosen_df["1_pred"] == chosen_nsub) | (chosen_df["2_pred"] == chosen_nsub) | (
        #             chosen_df["3_pred"] == chosen_nsub)]
        a = chosen_df[(chosen_df["1_pred"] == chosen_nsub) | (chosen_df["2_pred"] == chosen_nsub)]
        multi_dict[chosen_nsub] = a.shape[0]
    frequency_summary = pd.DataFrame.from_dict(overall_dict, orient='index')
    frequency_summary.rename({0: "overall_freq"}, axis=1, inplace=True)

    frequency_summary = pd.concat([frequency_summary, pd.DataFrame.from_dict(multi_dict, orient='index')], axis=1)
    frequency_summary.rename({0: "multi_freq"}, axis=1, inplace=True)

    overall_num_dict = {}
    for chosen_nsub in sorted(overall_proba_pred_ecod.nsub.unique().tolist()):
        chosen_df = overall_proba_pred_ecod[overall_proba_pred_ecod.nsub == chosen_nsub][["1_pred", "2_pred", "3_pred"]]
        overall_num_dict[chosen_nsub] = chosen_df.shape[0]
    overall_num_dict
    multi_num_dict = {}
    for chosen_nsub in sorted(multi_qs_tab.nsub.unique().tolist()):
        chosen_df = multi_qs_tab[multi_qs_tab.nsub == chosen_nsub][["1_pred", "2_pred", "3_pred"]]
        multi_num_dict[chosen_nsub] = chosen_df.shape[0]

    frequency_summary = pd.concat([frequency_summary, pd.DataFrame.from_dict(overall_num_dict, orient='index')], axis=1)
    frequency_summary.rename({0: "overall_count"}, axis=1, inplace=True)
    frequency_summary = pd.concat([frequency_summary, pd.DataFrame.from_dict(multi_num_dict, orient='index')], axis=1)
    frequency_summary.rename({0: "multi_count"}, axis=1, inplace=True)

    frequency_summary["overall_ratio"] = frequency_summary['overall_freq'] / frequency_summary['overall_count']
    frequency_summary["multi_ratio"] = frequency_summary["multi_freq"] / frequency_summary["multi_count"]

    frequency_summary[["overall_ratio", "multi_ratio"]].plot(kind="bar", xlabel="different qs",
                                                             ylabel="ratio of predictions from total amount",
                                                             title="overall top3 predictions by label compared to top3 predictions in multilabel clusters",
                                                             figsize=(9, 8))



if __name__ == "__main__":
    PATH = "/vol/ek/Home/orlyl02/working_dir/oligopred/mlp/esm_runs/5cv_sets_proba_models/"
    proba_pred_actual = pd.read_pickle(PATH + "proba_pred_actual.pkl")
    y_prob_with_overall_train = pd.read_pickle(PATH + "y_prob_with_overall_train.pkl")
    inv_map = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 10, 9: 12, 10: 14, 11: 24}
    multi_qs_clust, multi_qs_tab, single_qs_clust, overall_proba_pred, relevant_clusters = prepare_data(proba_pred_actual, y_prob_with_overall_train, inv_map, PATH)
    multi_cluster_summary_adjusted = cluster_analysis(multi_qs_clust, overall_proba_pred, PATH)
    multi_cluster_summary = cluster_analysis_not_adjusted(multi_qs_clust, overall_proba_pred, PATH, "multi")
    single_cluster_summary = cluster_analysis_not_adjusted(single_qs_clust, overall_proba_pred, PATH, "single")
    print(proba_pred_actual.head())
