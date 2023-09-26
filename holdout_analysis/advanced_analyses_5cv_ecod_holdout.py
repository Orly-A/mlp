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
from scipy.stats import ranksums

"""
This script is used to analyze the results of the mlp model, for the hold-out set!!! using the esm lm and the 0.3 coverage.
I started with the script working_dir/oligopred/mlp/advanced_analyses_5cv.py, 
but since we want to use ecod f_id instead of the representative, I opened a new script.
"""

def top2_analysis(holdout_set, y_prob, PATH, inv_map):
    # currently seems ok!! :)
    y_prob.rename(columns=inv_map, inplace=True)
    y_prob.reset_index(inplace=True, drop=True)
    holdout_set.reset_index(inplace=True, drop=True)
    print(y_prob.columns)
    y_prob_with_holdout = pd.concat([holdout_set, y_proba_df], axis=1)
    y_prob_with_holdout = y_prob_with_holdout[['nsub', 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 24, 'code', 'representative']]
    convert_dict = {"nsub": float, 1: float, 2: float, 3: float, 4: float, 5: float, 6: float, 7: float, 8: float,
                    10: float, 12: float, 14: float, 24: float}
    y_prob_with_holdout = y_prob_with_holdout.astype(convert_dict)
    pickle.dump(y_prob_with_holdout, open(PATH + "y_prob_with_holdout.pkl", "wb"))

    proba_pred_actual = pd.concat((y_prob_with_holdout[[1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 24]]
                                   .apply(lambda x: x.nlargest(3).index, axis=1, result_type='expand'),
                                   y_prob_with_holdout.nsub.astype(int)), axis=1)

    proba_pred_actual['top_2'] = np.where(proba_pred_actual[1] == proba_pred_actual["nsub"], proba_pred_actual[1], proba_pred_actual[0])
    top2_bal_acc = round(metrics.balanced_accuracy_score(proba_pred_actual['nsub'].astype(int), proba_pred_actual['top_2'].astype(int)), 3)
    # gen_con_mat_and_fig(metrics.confusion_matrix(proba_pred_actual['nsub'].astype(int), proba_pred_actual['top_2'].astype(int)), dest_path,
    #                  ("top_2_confusion_matrix Adjusted_balanced_accuracy: " + str(top2_bal_acc)), inv_map)
    regular_bal_acc = round(metrics.balanced_accuracy_score(proba_pred_actual['nsub'].astype(int), proba_pred_actual[0].astype(int)), 3)
    # gen_con_mat_and_fig(metrics.confusion_matrix(proba_pred_actual['nsub'].astype(int), proba_pred_actual[0].astype(int)), dest_path,
    #                    ("regular_confusion_matrix Adjusted_balanced_accuracy: " + str(regular_bal_acc)), inv_map)
    proba_pred_actual.rename({0: "1_pred", 1: "2_pred", 2: "3_pred"}, axis=1, inplace=True)
    return proba_pred_actual, y_prob_with_holdout

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
    return overall_proba_pred_ecod


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
    a.figure.savefig(PATH + "different_oligomeric_states_per_cluster_f_id.png")
    b = overall_proba_pred_ecod.groupby("f_id").nunique("code").groupby("code").size().plot\
        (kind='bar', color="maroon", figsize=[20,7], fontsize=10, log=True, grid=False,
         title="number of different pdbs in each ECOD f_id", xlabel="number of unique protein sequences", ylabel="number of clusters")
    b.figure.savefig(PATH + "number_of_unique_pdbs_per_cluster_f_id.png")
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
    plt.savefig(PATH + "qs_distribution_ecod_f_id.png")
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
        # chosen_df = overall_proba_pred_ecod[overall_proba_pred_ecod.nsub == chosen_nsub][["1","2","3","4","5","6","7","8","10","12","14","24"]]
        chosen_df = overall_proba_pred_ecod[overall_proba_pred_ecod.nsub == chosen_nsub][[1,2,3,4,5,6,7,8,10,12,14,24]]
        fig, axes = plt.subplots()
        sns.violinplot(data=chosen_df,ax=axes, scale="count", cut=0, bw=0.3)
        plt.title("'Confidence' in each label when actual label is " + str(chosen_nsub))
        plt.savefig(PATH + "violin_confidence_in_each_label_when_actual_label_is_" + str(chosen_nsub) + ".png")
        plt.close()


    for chosen_nsub in sorted(overall_proba_pred_ecod.nsub.unique().tolist()):
        chosen_df = overall_proba_pred_ecod[overall_proba_pred_ecod.nsub == chosen_nsub][[int(chosen_nsub)]]
        chosen_df_multi = multi_qs_tab[multi_qs_tab.nsub == chosen_nsub][[int(chosen_nsub)]]
        chosen_df_single = single_qs_tab[single_qs_tab.nsub == chosen_nsub][[int(chosen_nsub)]]
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
    plt.savefig(PATH + "success_of_each_method_by_qs.png")
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

def complementarity_esm_pisa(overall_set_proba_pisa, PATH):
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
    print(overall_set_proba_pisa.groupby(["nsub", "complementarity_esm_pisa"]).count().reset_index().columns)
    grouped_nsub_comp = overall_set_proba_pisa.groupby(["nsub", "complementarity_esm_pisa"]).count().reset_index()[
        ["nsub", "complementarity_esm_pisa", "code"]]
    grouped_nsub_comp.rename({"code": "counts"}, axis=1, inplace=True)
    pivoted_grouped_nsub_comp = grouped_nsub_comp.pivot(columns="complementarity_esm_pisa", index="nsub", values='counts')
    pivoted_grouped_nsub_comp = pivoted_grouped_nsub_comp[["both_true", "only_pisa", "only_esm", "both_false"]]
    pivoted_large = pivoted_grouped_nsub_comp.loc[pivoted_grouped_nsub_comp.index.isin([1, 2, 3, 4, 6])]
    pivoted_large.plot.bar(stacked=True, color=["lightgray", "plum", "midnightblue", "dimgray"])
    plt.xlabel("qs", fontsize=15)
    plt.ylabel("percentage of success", fontsize=15)
    plt.figure(figsize=(12, 10))
    # plt.legend().remove()
    plt.savefig(PATH + "complementarity_esm_pisa_large.png")
    plt.close()
    pivoted_small = pivoted_grouped_nsub_comp.loc[pivoted_grouped_nsub_comp.index.isin([5, 7, 8, 10, 12, 14, 24])]
    pivoted_small.plot.bar(stacked=True, color=["lightgray", "plum", "midnightblue", "dimgray"])
    plt.xlabel("qs", fontsize=15)
    plt.ylabel("percentage of success", fontsize=15)
    plt.figure(figsize=(12, 10))
    plt.savefig(PATH + "complementarity_esm_pisa_small.png")
    plt.close()


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


def acc_by_nsub_multi_labels_clust(overall_proba_pred_ecod, multi_qs_tab, single_qs_tab, PATH):
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
    plt.savefig(PATH + "accuracy_each_label_for_single_multi_clust.png")
    plt.close()
    size_nsub_df[["only_single", "multi_most", "multi_small"]].plot(kind="bar", color=["gold", "lightgreen", "darkgreen"], width=0.85, xlabel="qs", ylabel="accuracy", figsize=(12, 8), log=True)
    plt.ylim(0,1000)
    plt.savefig(PATH + "count_each_label_for_single_multi_clust.png")




def proba_dist_right_wrong(overall_proba_pred_ecod, PATH):
    distribution_df = pd.DataFrame(index=range(overall_proba_pred_ecod.shape[0]))
    for chosen_nsub in sorted(overall_proba_pred_ecod.nsub.unique().tolist()):
        chosen_df = overall_proba_pred_ecod[overall_proba_pred_ecod["1_pred"] == chosen_nsub][["nsub", "1_pred", int(chosen_nsub)]]
        chosen_df_true = chosen_df[chosen_df["nsub"] == chosen_nsub]
        chosen_df_wrong = chosen_df[chosen_df["nsub"] != chosen_nsub]
        distribution_df[str(int(chosen_nsub)) + "_correct"] = chosen_df_true[int(chosen_nsub)]
        distribution_df[str(int(chosen_nsub)) + "_wrong"] = chosen_df_wrong[int(chosen_nsub)]
        print(ranksums(chosen_df_true[int(chosen_nsub)], chosen_df_wrong[int(chosen_nsub)], alternative='greater'), "wilcoxon for " + str(int(chosen_nsub)))
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


def generate_final_tabs():
    holdout_overall_proba_pred_ecod = overall_proba_pred_ecod.copy()
    train_overall_proba_pred_ecod = pd.read_csv("/vol/ek/Home/orlyl02/working_dir/oligopred/mlp/esm_runs/5cv_sets_proba_models/overall_proba_pred_ecod.csv", sep="\t")
    train_overall_proba_pred_ecod["set"] = "train_set"
    holdout_overall_proba_pred_ecod["set"] = "hold_out_set"
    train_overall_proba_pred_ecod.drop("Unnamed: 0", axis=1, inplace=True)
    holdout_overall_proba_pred_ecod.rename({1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 10: "10", 12: "12", 14: "14", 24: "24"}, axis=1, inplace=True)
    all_model_data = pd.concat([train_overall_proba_pred_ecod, holdout_overall_proba_pred_ecod], axis=0)
    all_model_data.to_csv(PATH + "all_model_data.csv", sep='\t')
    esm_tab = pd.read_pickle("/vol/ek/Home/orlyl02/working_dir/oligopred/esmfold_prediction/esm_tab.pkl")
    esm_tab["set_name"] = esm_tab.apply(get_set_name, args=(train_overall_proba_pred_ecod, holdout_overall_proba_pred_ecod,), axis=1)
    esm_tab.to_csv(PATH + "embed_tab_with_sets.csv")

def get_set_name(a, train_overall_proba_pred_ecod, holdout_overall_proba_pred_ecod):
    # this worked for me in the debugger, without the need to send the train and holdout tabs, since they were global vars there.
    # that would be the way to go if this doesn't work in the flow of the script
    # print(a)
    # print(a["code"])
    if a["code"] in (train_overall_proba_pred_ecod.code.to_list()):
        return "train_included_in_model"
    elif a["code"] in (holdout_overall_proba_pred_ecod.code.to_list()):
        return "hold_out_set"
    else:
        return "train_not_used"


if __name__ == "__main__":

    PATH = "/vol/ek/Home/orlyl02/working_dir/oligopred/mlp/holdout_analysis/"
    overall_train_set = pd.read_pickle(PATH + "overall_train_c03_final.pkl")
    holdout_set = pd.read_pickle(PATH + "hold_out_set_c03_final.pkl")
    with open(PATH + "proba_8020_downsample3_final.pkl", "rb") as f:
        y_prob = pickle.load(f)
    y_proba_df = pd.DataFrame(y_prob)
    inv_map = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 10, 9: 12, 10: 14, 11: 24}
    proba_pred_actual, y_prob_with_holdout = top2_analysis(holdout_set, y_proba_df, PATH, inv_map)
    # y_prob_with_holdout = pd.read_pickle(PATH + "y_prob_with_holdout.pkl")
    overall_proba_pred = pd.concat([y_prob_with_holdout, proba_pred_actual[['1_pred', '2_pred', '3_pred', 'top_2']]], axis=1)
    overall_proba_pred_ecod = ecod(overall_proba_pred)

    multi_qs_clust, multi_qs_tab, single_qs_clust, single_qs_tab, relevant_clusters = initial_analysis(overall_proba_pred_ecod, PATH)
    multi_adjusted_cluster_summary_df = cluster_analysis(multi_qs_clust, overall_proba_pred_ecod, PATH)
    multi_NOTadjusted_cluster_summary_df = cluster_analysis_not_adjusted(multi_qs_clust, overall_proba_pred_ecod, PATH, "multi")
    single_cluster_summary_df = cluster_analysis_not_adjusted(single_qs_clust, overall_proba_pred_ecod, PATH, "single")
    num_of_qs_vs_predicted_num_of_qs(overall_proba_pred_ecod, PATH)
    overall_set_proba_pisa = get_pisa_eppic_annot(overall_proba_pred_ecod, relevant_clusters, PATH)
    # the ecod table is generated in the previous script, we just load it here, in the beginning of this function
    # probabilities_analyses(overall_proba_pred_ecod, PATH, multi_qs_tab)
    get_pisa_eppic_annot(overall_proba_pred_ecod, relevant_clusters, PATH)
    complementarity_esm_pisa(overall_set_proba_pisa, PATH)
    acc_by_nsub_multi_labels_clust(overall_proba_pred_ecod, multi_qs_tab, single_qs_tab, PATH)
    proba_dist_right_wrong(overall_proba_pred_ecod, PATH)
    with open(PATH + 'overall_proba_pred_ecod_holdout.pkl', 'wb') as f:
        pickle.dump(overall_proba_pred_ecod, f)
    print("finished")

