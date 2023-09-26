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


PATH = "/vol/ek/Home/orlyl02/working_dir/oligopred/embed_transfer/"
# for use with esm embeddings
cov_PATH = "/vol/ek/Home/orlyl02/working_dir/oligopred/embed_transfer/esm_embeds/"

# cov_PATH = "/vol/ek/Home/orlyl02/working_dir/oligopred/embed_transfer/transfer_cov03/"
SAVE_PATH = cov_PATH
NUM_CV = 5


def get_data():
    # overall_train_set = pd.read_pickle(cov_PATH + "train_set_c0.3.pkl")
    #for esm embeddings
    # overall_train_set = pd.read_pickle("/vol/ek/Home/orlyl02/working_dir/oligopred/esmfold_prediction/train_set_c0.3.pkl")
    # overall_train_set.reset_index(drop=True, inplace=True)

    train_set = pd.read_pickle("/vol/ek/Home/orlyl02/working_dir/oligopred/esmfold_prediction/train_set_c0.3.pkl")
    test_set = pd.read_pickle("/vol/ek/Home/orlyl02/working_dir/oligopred/esmfold_prediction/hold_out_set_c0.3.pkl")
    return train_set, test_set

def get_matrices(train_set):
    alnRes_mat = pd.read_pickle(PATH + "alnRes_blast1_mat.pkl")
    cosine_sim_df = pd.read_pickle(cov_PATH + "cosine_sim_df_c0.3_esm.pkl")
    pdbs_not_in_blast = [x for x in cosine_sim_df.index.to_list() if x not in alnRes_mat.index.to_list()]
    pdbs_in_holdout = [x for x in alnRes_mat.index.to_list() if x not in train_set.code.to_list()]
    pdbs_to_remove = pdbs_not_in_blast + pdbs_in_holdout
    alnRes_mat.drop(pdbs_to_remove, axis=0, errors='ignore', inplace=True)
    alnRes_mat.drop(pdbs_to_remove, axis=1, errors='ignore', inplace=True)
    cosine_sim_df.drop(pdbs_to_remove, axis=0, errors='ignore', inplace=True)
    cosine_sim_df.drop(pdbs_to_remove, axis=1, errors='ignore', inplace=True)
    return alnRes_mat, cosine_sim_df, pdbs_to_remove


def get_label_score_loop_seq(family, ref_mat, train_set):
    for index, row in family.iterrows():
        pdb_code = row["code"]
        ref_mat_copy = ref_mat.copy()
        ref_mat_copy.drop(pdb_code, axis=1, errors='ignore', inplace=True)
        ref = ref_mat_copy.loc[pdb_code].idxmax()
        label = float(train_set[train_set["code"] == ref].nsub)
        family.loc[index, "alnRes_code_homolog"] = ref
        family.loc[index, "alnRes_pred_homolog"] = label
        family.loc[index, "alnRes_max_score_homolog"] = ref_mat_copy.loc[pdb_code].max()
    return family


def get_label_score_loop_cos(family, ref_mat, train_set):
    for index, row in family.iterrows():
        pdb_code = row["code"]
        ref_mat_copy = ref_mat.copy()
        ref_mat_copy.drop(pdb_code, axis=1, errors='ignore', inplace=True)
        ref = ref_mat_copy.loc[pdb_code].idxmax()
        label = float(train_set[train_set["code"] == ref].nsub)
        family.loc[index, "cos_code_homolog"] = ref
        family.loc[index, "cos_pred_homolog"] = label
        family.loc[index, "cos_max_score_homolog"] = ref_mat_copy.loc[pdb_code].max()
    return family



if __name__ == "__main__":
    overall_proba_pred_ecod = pd.read_csv("/vol/ek/Home/orlyl02/working_dir/oligopred/mlp/esm_runs/5cv_sets_proba_models/overall_proba_pred_ecod.csv", sep="\t")
    family = overall_proba_pred_ecod[overall_proba_pred_ecod["f_id"] == "2492.1.1.5"][["f_name", "code", "nsub", "1_pred"]]
    train_set, test_set = get_data()
    alnRes_mat, cosine_sim_df, pdbs_to_remove = get_matrices(train_set)
    family = get_label_score_loop_seq(family, alnRes_mat , train_set)
    family = get_label_score_loop_cos(family, cosine_sim_df, train_set)

    print(family)

