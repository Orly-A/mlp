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
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import roc_auc_score
from sklearn.metrics import (precision_recall_curve, PrecisionRecallDisplay)


"""
This script is used to analyze the results of the mlp model, for all 5 cv, using the esm lm and the 0.3 coverage.
I started with the script working_dir/oligopred/mlp/advanced_analyses_5cv.py, 
but since we want to use ecod f_id instead of the representative, I opened a new script.
"""

def initial_analysis(overall_proba_pred_ecod, PATH):
    multi_qs_clust = overall_proba_pred_ecod.groupby("f_id").nunique()[overall_proba_pred_ecod.groupby("f_id")
                                                                                       .nunique().nsub > 1].sort_values("nsub")
    relevant_clusters = multi_qs_clust[["nsub"]]
    multi_qs_tab = overall_proba_pred_ecod[overall_proba_pred_ecod["f_id"].isin(relevant_clusters.index)]
    regular_bal_acc = round(metrics.balanced_accuracy_score(overall_proba_pred_ecod['nsub'].astype(int), overall_proba_pred_ecod["1_pred"].astype(int), adjusted=True), 3)
    gen_con_mat_and_fig(metrics.confusion_matrix(overall_proba_pred_ecod['nsub'].astype(int), overall_proba_pred_ecod["1_pred"].astype(int)), PATH,
                       ("regular_confusion_matrix Adjusted_balanced_accuracy: " + str(regular_bal_acc)), inv_map)
    multi_qs_bal_acc = round(metrics.balanced_accuracy_score(multi_qs_tab['nsub'].astype(int), multi_qs_tab["1_pred"].astype(int), adjusted=True), 3)
    gen_con_mat_and_fig(metrics.confusion_matrix(multi_qs_tab['nsub'].astype(int), multi_qs_tab["1_pred"].astype(int)), PATH,
                          ("multi_qs_confusion_matrix_f_id Adjusted_balanced_accuracy: " + str(multi_qs_bal_acc)), inv_map)
    single_qs_clust = overall_proba_pred_ecod.groupby("f_id").nunique()[overall_proba_pred_ecod.groupby("f_id")
                                                                                   .nunique().nsub == 1].sort_values("nsub")
    single_qs_tab = overall_proba_pred_ecod[~overall_proba_pred_ecod["f_id"].isin(relevant_clusters.index)]

    a = overall_proba_pred_ecod.groupby("f_id").nunique("nsub").groupby("nsub").size().plot\
        (kind='bar', grid=False, log=True, color="maroon", fontsize=10,
         title="different oligomeric states per ECOD f_id", xlabel="different oligomeric states", ylabel="number of clusters")
    # a.figure.savefig(PATH + "different_oligomeric_states_per_cluster_f_id.png")
    b = overall_proba_pred_ecod.groupby("f_id").nunique("code").groupby("code").size().plot\
        (kind='bar', color="maroon", figsize=[20,7], fontsize=10, log=True, grid=False,
         title="number of different pdbs in each ECOD f_id", xlabel="number of unique protein sequences", ylabel="number of clusters")
    # b.figure.savefig(PATH + "number_of_unique_pdbs_per_cluster_f_id.png")
    b.clear()

    return multi_qs_clust, multi_qs_tab, single_qs_clust, single_qs_tab, relevant_clusters


def cluster_analysis(multi_qs_clust, overall_proba_pred_ecod, PATH):
    cluster_summary_df = pd.DataFrame(columns=["f_id", "num_of_qs", "nsub_classes", "num_of_pdbs", "adj_bal_ac", "adj_bal_ac_top2"])
    for rep in multi_qs_clust.index.to_list():
        tab = overall_proba_pred_ecod[overall_proba_pred_ecod["f_id"] == rep]
        bal_ac = round(metrics.balanced_accuracy_score(tab['nsub'].astype(int), tab["1_pred"].astype(int), adjusted=True), 3)
        bal_ac_top2 = round(metrics.balanced_accuracy_score(tab['nsub'].astype(int), tab["top_2"].astype(int), adjusted=True), 3)
        cluster_summary_dict = {}
        cluster_summary_dict["f_id"] = str(rep)
        cluster_summary_dict["num_of_qs"] = multi_qs_clust.loc[rep, "nsub"]
        cluster_summary_dict["nsub_classes"] = list(tab.nsub.unique())
        cluster_summary_dict["num_of_pdbs"] = tab.shape[0]
        cluster_summary_dict["adj_bal_ac"] = bal_ac
        cluster_summary_dict["adj_bal_ac_top2"] = bal_ac_top2
        cluster_summary_df = cluster_summary_df.append(cluster_summary_dict, ignore_index=True)
    cluster_summary_df.to_csv(PATH + "multi_qs_clusters_summary_f_id.csv", index=False, sep="\t")
    cluster_summary_df.to_pickle(PATH + "multi_qs_clusters_summary_f_id.pkl")
    return cluster_summary_df



def cluster_analysis_not_adjusted(qs_clust, overall_proba_pred_ecod, PATH, name):
    cluster_summary_df = pd.DataFrame(columns=["f_id", "num_of_qs", "nsub_classes", "num_of_pdbs", "bal_ac", "bal_ac_top2"])
    for rep in qs_clust.index.to_list():
        tab = overall_proba_pred_ecod[overall_proba_pred_ecod["f_id"] == rep]
        bal_ac = round(metrics.balanced_accuracy_score(tab['nsub'].astype(int), tab["1_pred"].astype(int)), 3)
        bal_ac_top2 = round(metrics.balanced_accuracy_score(tab['nsub'].astype(int), tab["top_2"].astype(int)), 3)
        cluster_summary_dict = {}
        cluster_summary_dict["f_id"] = str(rep)
        cluster_summary_dict["num_of_qs"] = qs_clust.loc[rep, "nsub"]
        cluster_summary_dict["nsub_classes"] = list(tab.nsub.unique())
        cluster_summary_dict["num_of_pdbs"] = tab.shape[0]
        cluster_summary_dict["bal_ac"] = bal_ac
        cluster_summary_dict["bal_ac_top2"] = bal_ac_top2
        cluster_summary_df = cluster_summary_df.append(cluster_summary_dict, ignore_index=True)
    cluster_summary_df.to_csv(PATH + name + "_qs_clusters_summary_not_adjusted_f_id.csv", index=False, sep="\t")
    cluster_summary_df.to_pickle(PATH + name + "_qs_clusters_summary_not_adjusted_f_id.pkl")
    bal_ac_vs_num_qs_per_clus(cluster_summary_df, name, PATH)
    return cluster_summary_df



def gen_con_mat_and_fig(mat, dest_path, title, inv_map):
    mat_df = pd.DataFrame(mat)
    #check this is the correct mapping
    # inv_map = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 8, 7: 10, 8: 12, 9: 24}
    mat_df.rename(inv_map, axis=1, inplace=True)
    mat_df.rename(inv_map, inplace=True)
    mat_df_per = mat_df.div(mat_df.sum(axis=1), axis=0).round(2)
    s = sns.heatmap(mat_df_per, annot=True, fmt='g', cmap="BuPu", vmin=0, vmax=1, annot_kws={"size": 8})
    s.figure.set_size_inches(8, 6)
    s.set(xlabel='Prediction', ylabel='Actual_lables', title=title)
    plt.savefig(dest_path + title.split(" ")[0] + ".png")
    # plt.show()
    plt.close()
    s.clear()


def num_of_qs_vs_predicted_num_of_qs(overall_proba_pred_ecod, PATH):
    num_qs_num_pred_qs = overall_proba_pred_ecod.groupby("f_id").nunique().sort_values("nsub")[["nsub", "code", "1_pred"]]

    num_qs_num_pred_qs = num_qs_num_pred_qs[num_qs_num_pred_qs.code != 1]
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
    qs_dist_pivot_percent.plot.bar(stacked=True, figsize=(9, 8), xlabel="different qs in cluster", ylabel="total clusters", title="num_of_qs_vs_predicted_num_of_qs", fontsize=10, cmap="viridis_r")
    legend = plt.legend(title="num predicted qs", loc='upper right', fontsize='small', fancybox=True)#.remove()
    # plt.savefig(PATH + "qs_distribution_ecod_f_id.png")
    plt.close()


def bal_ac_vs_num_qs_per_clus(cluster_summary, name, PATH):
    pdbs_vs_ac = cluster_summary.groupby("num_of_pdbs").mean()["bal_ac"].reset_index()
    cluster_count = cluster_summary.groupby("num_of_pdbs").count().reset_index()[["num_of_pdbs", "f_id"]]
    pdbs_vs_ac_with_count = cluster_count.merge(pdbs_vs_ac, on="num_of_pdbs")
    pdbs_vs_ac.plot(x="num_of_pdbs", y="bal_ac", kind="bar", figsize=(9, 8), title=(name + "qs in cluster balanced accuracy vs number of pdbs per cluster"))
    plt.close()

def probabilities_analyses(overall_proba_pred_ecod, PATH, multi_qs_tab):
    # probability of each class by cluster and by nsub within
    # y_prob_with_overall_train.groupby(["representative", "nsub"]).mean()
    # biggest_clusters = y_prob_with_overall_train.groupby(["representative", "nsub"]).mean().groupby('representative').size().sort_values()[-3:-1].index.values
    # y_prob_with_overall_train.groupby(["representative", "nsub"]).mean().query("representative in @biggest_clusters").groupby(["representative", "nsub"]).plot.bar()
    #boxplots from the initial analysis with Matan
    # for chosen_nsub in sorted(overall_proba_pred_ecod.nsub.unique().tolist()):
    #     chosen_df = overall_proba_pred_ecod[overall_proba_pred_ecod.nsub == chosen_nsub][[1,2,3,4,5,6,7,8,10,12,14,24]]
    #     chosen_df.plot.box(title="'Confidence' in each label when actual label is " + str(chosen_nsub))
    #     plt.savefig(PATH + "confidence_in_each_label_when_actual_label_is_" + str(chosen_nsub) + ".png")
    #     plt.close()
    for chosen_nsub in sorted(overall_proba_pred_ecod.nsub.unique().tolist()):
        chosen_df = overall_proba_pred_ecod[overall_proba_pred_ecod.nsub == chosen_nsub][["1","2","3","4","5","6","7","8","10","12","14","24"]]
        fig, axes = plt.subplots()
        sns.violinplot(data=chosen_df,ax=axes, scale="count", cut=0, bw=0.3)
        plt.title("'Confidence' in each label when actual label is " + str(chosen_nsub))
        plt.savefig(PATH + "_violin_confidence_in_each_label_when_actual_label_is_" + str(chosen_nsub) + ".png")
        plt.close()



    for chosen_nsub in sorted(overall_proba_pred_ecod.nsub.unique().tolist()):
        chosen_df = overall_proba_pred_ecod[overall_proba_pred_ecod.nsub == chosen_nsub][[str(int(chosen_nsub))]]
        chosen_df_multi = multi_qs_tab[multi_qs_tab.nsub == chosen_nsub][[str(int(chosen_nsub))]]
        chosen_df_single = single_qs_tab[single_qs_tab.nsub == chosen_nsub][[str(int(chosen_nsub))]]
        chosen_df_concat = pd.concat([chosen_df, chosen_df_multi, chosen_df_single], axis=1)
        chosen_df_concat.columns.values[0] = "overall"
        chosen_df_concat.columns.values[1] = "within_multi_labels_clusters"
        chosen_df_concat.columns.values[2] = "within_single_labels_clusters"
        plt.hist(chosen_df_concat, color=['gray', 'darkgreen', 'darkorchid'], bins=20)
        plt.xlabel("probability")
        plt.ylabel("counts")
        plt.legend(["overall", "within_multi_labels_clusters", "within_single_labels_clusters"])
        plt.title("confidence_in_" + str(chosen_nsub) + "_when_actual_label_is_" + str(chosen_nsub))
        plt.savefig(PATH + "confidence_in_" + str(chosen_nsub) + "_when_actual_label_is_" + str(chosen_nsub) + ".png")
        # plt.show()
        plt.close()


def get_pisa_eppic_annot(overall_proba_pred_ecod, relevant_clusters, PATH):
    overall_qsbio = pd.read_csv("/vol/ek/Home/orlyl02/working_dir/oligopred/QSbio_PiQSi_annotations_V6_2020.csv", error_bad_lines=False, low_memory=False, skiprows=21)
    qsbio_relevant_rows = overall_qsbio[overall_qsbio["code"].isin(overall_proba_pred_ecod.code.to_list())]
    qsbio_relevant_rows = qsbio_relevant_rows[["code", 'sym', 'PISA_identical', 'EPPIC_identical']]
    overall_set_proba_pisa = overall_proba_pred_ecod.merge(qsbio_relevant_rows, on="code", how="left")
    overall_set_proba_pisa["esm_identical"] = np.where(overall_set_proba_pisa["1_pred"] == overall_set_proba_pisa["nsub"], 1, 0)
    overall_set_proba_pisa["PISA_identical"] = pd.to_numeric(overall_set_proba_pisa["PISA_identical"], errors='coerce', downcast="integer")
    overall_set_proba_pisa["EPPIC_identical"] = pd.to_numeric(overall_set_proba_pisa["EPPIC_identical"], errors='coerce', downcast="integer")
    overall_set_proba_pisa["1_pred"] = pd.to_numeric(overall_set_proba_pisa["1_pred"], errors='coerce', downcast="integer")
    overall_set_proba_pisa["2_pred"] = pd.to_numeric(overall_set_proba_pisa["2_pred"], errors='coerce', downcast="integer")
    overall_set_proba_pisa["3_pred"] = pd.to_numeric(overall_set_proba_pisa["3_pred"], errors='coerce', downcast="integer")
    method_comp = overall_set_proba_pisa[["PISA_identical", "EPPIC_identical", "esm_identical", "nsub", "code", "f_id"]]
    method_comp_grouped_nsub = method_comp.groupby("nsub").sum()
    method_comp_grouped_nsub_counts = method_comp_grouped_nsub.merge(method_comp.groupby("nsub").count()["f_id"], on="nsub", how="left")
    method_comp_grouped_nsub_counts.rename({"f_id": "count"}, axis=1, inplace=True)
    method_comp_grouped_nsub_counts_per = method_comp_grouped_nsub_counts.copy()
    method_comp_grouped_nsub_counts_per["PISA_identical"] = method_comp_grouped_nsub_counts["PISA_identical"]/method_comp_grouped_nsub_counts["count"]
    method_comp_grouped_nsub_counts_per["EPPIC_identical"] = method_comp_grouped_nsub_counts["EPPIC_identical"]/method_comp_grouped_nsub_counts["count"]
    method_comp_grouped_nsub_counts_per["esm_identical"] = method_comp_grouped_nsub_counts["esm_identical"]/method_comp_grouped_nsub_counts["count"]
    method_comp_grouped_nsub_counts_per[["PISA_identical", "EPPIC_identical", "esm_identical"]].plot(kind="bar", figsize=(12, 8),
                                                                                                           title="success of each method by qs",
                                                                                                           xlabel="qs",
                                                                                                           ylabel="percentage of success",
                                                                                                           width=0.85,
                                                                                                           fontsize=10,
                                                                                                           ylim=([0, 1]),
                                                                                                           color=["plum", "peachpuff", "midnightblue"])
    # plt.savefig(PATH + "success_of_each_method_by_qs.png")
    plt.close()
    multi_method_comp = method_comp[method_comp["f_id"].isin(relevant_clusters.index)]
    multi_method_comp_grouped_nsub = multi_method_comp.groupby("nsub").sum()
    multi_method_comp_grouped_nsub_counts = multi_method_comp_grouped_nsub.merge(multi_method_comp.groupby("nsub").count()["f_id"], on="nsub", how="left")
    multi_method_comp_grouped_nsub_counts.rename({"f_id": "count"}, axis=1, inplace=True)
    multi_method_comp_grouped_nsub_counts_per = multi_method_comp_grouped_nsub_counts.copy()
    multi_method_comp_grouped_nsub_counts_per["PISA_identical"] = multi_method_comp_grouped_nsub_counts["PISA_identical"]/method_comp_grouped_nsub_counts["count"]
    multi_method_comp_grouped_nsub_counts_per["EPPIC_identical"] = multi_method_comp_grouped_nsub_counts["EPPIC_identical"]/method_comp_grouped_nsub_counts["count"]
    multi_method_comp_grouped_nsub_counts_per["esm_identical"] = multi_method_comp_grouped_nsub_counts["esm_identical"]/method_comp_grouped_nsub_counts["count"]
    multi_method_comp_grouped_nsub_counts_per[['PISA_identical', 'EPPIC_identical', 'esm_identical']].plot(kind="bar",
                                                                                                           figsize=(12, 8),
                                                                                                           title="success of each method by qs",
                                                                                                           xlabel="qs",
                                                                                                           ylabel="percentage of success",
                                                                                                           width=0.85,
                                                                                                           fontsize=10,
                                                                                                           ylim=([0, 1]),
                                                                                                           color=["plum","peachpuff","midnightblue"])
    plt.legend().remove()
    # plt.savefig(PATH + "success_of_each_method_by_qs_for_multiclass_clusters_f_id.png")
    plt.close()
    return overall_set_proba_pisa

def complementarity_esm_pisa(overall_set_proba_pisa):
    both_false = (overall_set_proba_pisa["PISA_identical"] == overall_set_proba_pisa["esm_identical"]) & (
                overall_set_proba_pisa["esm_identical"] == 0)
    both_true = (overall_set_proba_pisa["PISA_identical"] == overall_set_proba_pisa["esm_identical"]) & (
                overall_set_proba_pisa["esm_identical"] == 1)
    only_esm = (overall_set_proba_pisa["PISA_identical"] != overall_set_proba_pisa["esm_identical"]) & (
                overall_set_proba_pisa["esm_identical"] == 1)
    only_pisa = (overall_set_proba_pisa["PISA_identical"] != overall_set_proba_pisa["esm_identical"]) & (
                overall_set_proba_pisa["esm_identical"] == 0)
    conditions = [both_false, both_true, only_esm, only_pisa]
    choices = ["both_false", "both_true", "only_esm", "only_pisa"]
    overall_set_proba_pisa["complementarity_esm_pisa"] = np.select(conditions, choices)
    grouped_nsub_comp = overall_set_proba_pisa.groupby(["nsub", "complementarity_esm_pisa"]).count().reset_index()[
        ["nsub", "complementarity_esm_pisa", "Unnamed: 0"]]
    grouped_nsub_comp.rename({"Unnamed: 0": "counts"}, axis=1, inplace=True)
    pivoted_grouped_nsub_comp = grouped_nsub_comp.pivot(columns="complementarity_esm_pisa", index="nsub", values='counts')
    pivoted_grouped_nsub_comp = pivoted_grouped_nsub_comp[["both_true", "only_pisa", "only_esm", "both_false"]]
    pivoted_large = pivoted_grouped_nsub_comp.loc[pivoted_grouped_nsub_comp.index.isin([1, 2, 3, 4, 6])]
    pivoted_small = pivoted_grouped_nsub_comp.loc[pivoted_grouped_nsub_comp.index.isin([5, 7, 8, 10, 12, 14, 24])]
    pivoted_large.plot.bar(stacked=True, color=["lightgray", "plum", "midnightblue", "dimgray"])
    plt.xlabel("qs", fontsize=15)
    plt.ylabel("percentage of success", fontsize=15)
    plt.figure(figsize=(12, 10))
    plt.legend().remove()
    pivoted_small.plot.bar(stacked=True, color=["lightgray", "plum", "midnightblue", "dimgray"])
    plt.xlabel("qs", fontsize=15)
    plt.ylabel("percentage of success", fontsize=15)
    plt.figure(figsize=(12, 10))
    plt.show()


def freq_by_label_in_multi_vs_all(overall_proba_pred_ecod, multi_qs_tab):
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
                                                             title="overall top2 predictions by label compared to top2 predictions in multilabel clusters",
                                                             figsize=(9, 8))
    plt.savefig(PATH + "overall_top2_predictions_by_label_compared_to_top2_predictions_in_f_id_multilabel_clusters.png")
    plt.close()


def nsub_stats(overall_train_set, overall_proba_pred_ecod, hold_out_set):
    tab_nsub_num = pd.concat([overall_train_set.drop("esm_embeddings", axis=1).groupby("nsub").nunique("code")["code"], overall_proba_pred_ecod.groupby("nsub").nunique("code")["code"], hold_out_set.drop("embeddings", axis=1).groupby("nsub").nunique("code")["code"]], axis=1)
    tab_nsub_num.columns.values[0] = "overall_train_set"
    tab_nsub_num.columns.values[1] = "used_for_training"
    tab_nsub_num.columns.values[2] = "hold_out_set"
    tab_nsub_num.plot(kind="bar", figsize=(10, 8), log=True, color=["black", "dimgray", "darkgray"], stacked=True)
    plt.title("number of different sequences for each qs")
    plt.savefig(PATH + "amount_seqs_per_qs.png")


def acc_by_nsub_multi_labels_clust(overall_proba_pred_ecod, multi_qs_tab, single_qs_tab):
    acc_by_nsub_dict = {}
    size_nsub = {}
    for chosen_nsub in sorted(overall_proba_pred_ecod.nsub.unique().tolist()):
        # get overall accuracy for nsub
        chosen_df = overall_proba_pred_ecod[overall_proba_pred_ecod.nsub == chosen_nsub][["nsub", "1_pred"]]
        accuracy_calc = round(metrics.accuracy_score(chosen_df['nsub'].astype(int), chosen_df["1_pred"].astype(int)), 2)
        # get accuracy for single label clusters
        chosen_df_single = single_qs_tab[single_qs_tab.nsub == chosen_nsub][["nsub", "1_pred"]]
        accuracy_calc_single = round(metrics.accuracy_score(chosen_df_single['nsub'].astype(int), chosen_df_single["1_pred"].astype(int)), 2)
        # get accuracy for multilabel clusters, when the chosen_nsub is the mjority and when it's the minority
        list_fid_chosen_nsub = multi_qs_tab[multi_qs_tab["nsub"] == chosen_nsub]["f_id"].unique().tolist()
        chosen_multi = multi_qs_tab[multi_qs_tab["f_id"].isin(list_fid_chosen_nsub)]
        a = chosen_multi.groupby(["f_id", "nsub"]).count().reset_index()[chosen_multi.groupby(["f_id", "nsub"]).count().reset_index()["nsub"] == chosen_nsub][["f_id", "code"]]
        b = chosen_multi.groupby(["f_id"]).count().reset_index()[["f_id", "code"]]
        a.rename({"code": "chosen"}, axis=1, inplace=True)
        b.rename({"code": "overall_count"}, axis=1, inplace=True)
        merged = a.merge(b, how="inner")
        merged["chosen_is_most"] = np.where((merged["chosen"].astype(int) > merged["overall_count"]/2), 1, np.nan)
        merged["chosen_is_small"] = np.where((merged["chosen"].astype(int) < merged["overall_count"]/2), 1, np.nan)
        subset_of_most = chosen_multi[chosen_multi["f_id"].isin(merged[merged["chosen_is_most"] == 1]["f_id"].to_list())]
        only_chosen_most = subset_of_most[subset_of_most["nsub"] == chosen_nsub]
        accuracy_calc_most = round(metrics.accuracy_score(only_chosen_most['nsub'].astype(int), only_chosen_most["1_pred"].astype(int)), 2)
        subset_of_small = chosen_multi[chosen_multi["f_id"].isin(merged[merged["chosen_is_small"] == 1]["f_id"].to_list())]
        only_chosen_small = subset_of_small[subset_of_small["nsub"] == chosen_nsub]
        accuracy_calc_small = round(metrics.accuracy_score(only_chosen_small['nsub'].astype(int), only_chosen_small["1_pred"].astype(int)), 2)
        # save all to a dictionary
        acc_by_nsub_dict[int(chosen_nsub)] = (accuracy_calc, accuracy_calc_single, accuracy_calc_most, accuracy_calc_small)
        acc_by_nsub_df = pd.DataFrame.from_dict(acc_by_nsub_dict, orient="index")
        acc_by_nsub_df.rename({0: "overall", 1: "only_single", 2: "multi_most", 3: "multi_small"}, axis=1, inplace=True)
        size_nsub[int(chosen_nsub)] = (chosen_df.shape[0], chosen_df_single.shape[0], only_chosen_most.shape[0], only_chosen_small.shape[0])
    acc_by_nsub_df = pd.DataFrame.from_dict(acc_by_nsub_dict, orient="index")
    acc_by_nsub_df.rename({0: "overall", 1: "only_single", 2: "multi_most", 3: "multi_small"}, axis=1, inplace=True)
    size_nsub_df = pd.DataFrame.from_dict(size_nsub, orient="index")
    size_nsub_df.rename({0: "overall", 1: "only_single", 2: "multi_most", 3: "multi_small"}, axis=1, inplace=True)
    acc_by_nsub_df.plot(kind="bar", color=["gray", "gold", "lightgreen", "darkgreen"], width=0.85, xlabel="qs", ylabel="accuracy", figsize=(12, 8), title="accuracy per qs for overall, single qs and multi-qs clusters")
    plt.show()


def proba_dist_right_wrong(overall_proba_pred_ecod):
    distribution_df = pd.DataFrame(index=range(overall_proba_pred_ecod.shape[0]))
    for chosen_nsub in sorted(overall_proba_pred_ecod.nsub.unique().tolist()):
        chosen_df = overall_proba_pred_ecod[overall_proba_pred_ecod["1_pred"] == chosen_nsub][["nsub", "1_pred", str(int(chosen_nsub))]]
        chosen_df_true = chosen_df[chosen_df["nsub"] == chosen_nsub]
        chosen_df_wrong = chosen_df[chosen_df["nsub"] != chosen_nsub]
        distribution_df[str(int(chosen_nsub)) + "_correct"] = chosen_df_true[str(int(chosen_nsub))]
        distribution_df[str(int(chosen_nsub)) + "_wrong"] = chosen_df_wrong[str(int(chosen_nsub))]
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.setp(sns.boxplot(data=distribution_df, palette=["darkturquoise", "darkorchid"]).get_xticklabels(), rotation=45)
    ax.set_ylabel("probability")
    ax.set_title("'Confidence' in predicted label when correct (green) or wrong (purple)")
    plt.savefig(PATH + "probability_each_label_for_true_and_false_positives.png")
    plt.close()

def calc_sym_differences(overall_set_proba_pisa):
    for qs in [4, 6, 8, 10, 12, 14]:
        tab = overall_set_proba_pisa[overall_set_proba_pisa["nsub"] == qs]
        print(qs)
        # print(tab["sym"].unique())
        tab_c = tab[tab["sym"] == str("C" + str(qs))]
        tab_d = tab[tab["sym"] == str("D" + str(int(qs / 2)))]
        # print(tab_c[["sym", "1_pred"]])
        # print(tab_d[["sym", "1_pred"]])
        correct_c = tab_c[tab_c["nsub"] == tab_c["1_pred"]]
        correct_d = tab_d[tab_d["nsub"] == tab_d["1_pred"]]
        print(round(correct_c.shape[0] / tab_c.shape[0], 3), "C" + str(qs))
        print(round(correct_d.shape[0] / tab_d.shape[0], 3), "D" + str(int(qs / 2)))

#########################################
### added for review analysis

def calc_reps(overall_proba_pred_ecod):
    rep_rows = overall_proba_pred_ecod[overall_proba_pred_ecod["code"] == overall_proba_pred_ecod["representative"]]
    bal_acc = round(metrics.balanced_accuracy_score(rep_rows['nsub'].astype(int), rep_rows["1_pred"].astype(int), adjusted=False), 3)
    f1 = round(metrics.f1_score(rep_rows['nsub'].astype(int), rep_rows["1_pred"].astype(int), average="weighted"), 3)
    ecod_rep_rows = overall_proba_pred_ecod.drop_duplicates(subset=["f_id"])
    ecod_bal_acc = round(metrics.balanced_accuracy_score(ecod_rep_rows['nsub'].astype(int), ecod_rep_rows["1_pred"].astype(int), adjusted=False), 3)
    ecod_f1 = round(metrics.f1_score(ecod_rep_rows['nsub'].astype(int), ecod_rep_rows["1_pred"].astype(int), average="weighted"), 3)

    with open('/vol/ek/Home/orlyl02/working_dir/oligopred/mlp/holdout_analysis/overall_proba_pred_ecod_holdout.pkl', 'rb') as f:
        overall_proba_pred_ecod_holdout = pickle.load(f)

    ecod_rep_results_dict = {}
    for col in overall_proba_pred_ecod.columns:
        a = overall_proba_pred_ecod.sort_values(by=col)
        ecod_rep_rows = a.drop_duplicates(subset=["f_id"])
        ecod_bal_acc = round(metrics.balanced_accuracy_score(ecod_rep_rows['nsub'].astype(int), ecod_rep_rows["1_pred"].astype(int), adjusted=False), 2)
        ecod_f1 = round(metrics.f1_score(ecod_rep_rows['nsub'].astype(int), ecod_rep_rows["1_pred"].astype(int), average="weighted"), 2)
        ecod_rep_results_dict[col]=[ecod_bal_acc, ecod_f1]
        columns = ["BA", "F1", "sorted_by"]
        ecod_rep_results = pd.DataFrame(columns=columns)
    print(ecod_rep_results_dict)
    ecod_rep_results = pd.DataFrame(ecod_rep_results_dict).transpose()
    ecod_rep_results.rename(columns={0: "BA", 1: "F1"}, inplace=True)
    ecod_rep_results.drop_duplicates(subset=("BA", "F1"), inplace=True)
    ecod_rep_results["BA"].mean()
    ecod_rep_results["BA"].std()
    ecod_rep_results["F1"].mean()
    ecod_rep_results["F1"].std()





def calc_more_measures(overall_proba_pred_ecod):
    # Accuracy, sensitivity, specificity, ROC curve, Precision-Recall curve
    accuracy = round(metrics.accuracy_score(overall_proba_pred_ecod['nsub'].astype(int), overall_proba_pred_ecod["1_pred"].astype(int)), 2)
    print(metrics.classification_report(overall_proba_pred_ecod['nsub'].astype(int), overall_proba_pred_ecod["1_pred"].astype(int)))
    #rocauc
    overall_proba_pred_ecod_maxval = pd.concat\
        ((overall_proba_pred_ecod.drop("Unnamed: 0", axis=1), overall_proba_pred_ecod[["1", "2", "3", "4", "5", "6", "7", "8", "10", "12", "14", "24"]].max(axis=1)), axis=1)
    overall_proba_pred_ecod_maxval.rename({0: "val_predicted"}, axis=1, inplace=True)
    overall_proba_pred_ecod["correct"] = np.where(overall_proba_pred_ecod["1_pred"] == overall_proba_pred_ecod["nsub"], 1, 0)
    y_onehot_test = overall_proba_pred_ecod["correct"]
    y_score = overall_proba_pred_ecod_maxval["val_predicted"]
    RocCurveDisplay.from_predictions(
        y_onehot_test.ravel(),
        y_score.ravel(),
        name="micro-average OvR",
        color="darkorange",
        #   plot_chance_level=True,
    )
    plt.axis("square")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Micro-averaged One-vs-Rest\nReceiver Operating Characteristic")
    plt.legend()
    plt.show()
    #ROC-AUC score - only macro and weighted
    y_score_roc_multi = overall_proba_pred_ecod[["1", "2", "3", "4", "5", "6", "7", "8", "10", "12", "14", "24"]]
    y_test = overall_proba_pred_ecod["nsub"]
    weighted_roc_auc_ovr = roc_auc_score(
        y_test,
        y_score_roc_multi,
        multi_class="ovr",
        average="weighted")
    print(f"weighted-averaged One-vs-Rest ROC AUC score:\n{weighted_roc_auc_ovr:.2f}")
    #precision recall curve
    precision, recall, thresholds = precision_recall_curve(y_onehot_test, y_score)
    disp = PrecisionRecallDisplay(precision=precision, recall=recall)
    disp.plot()
    plt.show()





if __name__ == "__main__":
    PATH = "/vol/ek/Home/orlyl02/working_dir/oligopred/mlp/esm_runs/5cv_sets_proba_models/"
    proba_pred_actual = pd.read_pickle(PATH + "proba_pred_actual.pkl")
    y_prob_with_overall_train = pd.read_pickle(PATH + "y_prob_with_overall_train.pkl")
    overall_proba_pred_ecod = pd.read_csv(PATH + "overall_proba_pred_ecod.csv", sep="\t")
    overall_train_set = pd.read_pickle("/vol/ek/Home/orlyl02/working_dir/oligopred/esmfold_prediction/train_set_c0.3.pkl")
    hold_out_set = pd.read_pickle("/vol/ek/Home/orlyl02/working_dir/oligopred/clustering/re_clust_c0.3/hold_out_set_c0.3.pkl")
    inv_map = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 10, 9: 12, 10: 14, 11: 24}
    multi_qs_clust, multi_qs_tab, single_qs_clust, single_qs_tab, relevant_clusters = initial_analysis(overall_proba_pred_ecod, PATH)
    multi_adjusted_cluster_summary_df = cluster_analysis(multi_qs_clust, overall_proba_pred_ecod, PATH)
    multi_NOTadjusted_cluster_summary_df = cluster_analysis_not_adjusted(multi_qs_clust, overall_proba_pred_ecod, PATH, "multi")
    single_cluster_summary_df = cluster_analysis_not_adjusted(single_qs_clust, overall_proba_pred_ecod, PATH, "single")
    num_of_qs_vs_predicted_num_of_qs(overall_proba_pred_ecod, PATH)
    overall_set_proba_pisa = get_pisa_eppic_annot(overall_proba_pred_ecod, relevant_clusters, PATH)
    # the ecod table is generated in the previous script, we just load it here, in the beginning of this function
    print("finished")

