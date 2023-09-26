import re
import os
import pickle
import sys
import datetime
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def open_score_files(args):
    # open the score files from pkl as df and concat them
    score_files = []
    for file_name in args:
        with open(file_name, 'rb') as f:
            score_files.append(pickle.load(f))
    path = args[0].rsplit("/", 1)[0]
    return pd.concat(score_files), path


def generate_figs(score_files, save_path):
    x_list = ['activation', 'alpha', 'batch_size', 'hidden_layer_sizes', 'learning_rate', 'learning_rate_init',
              'max_iter', 'n_iter_no_change', 'solver', 'tol']
    y_list = ["mean_f1_score_test", "mean_bal_acc_test"]
    for y in y_list:
        for x in x_list:
            fig, ax = plt.subplots()
            sns.boxplot(x=x, y=y, data=score_files)
            ax.set_title(x + " vs " + y)
            ax.set_xlabel(x)
            ax.set_ylabel(y)
            fig.savefig(save_path + "/" + x + "_vs_" + y + ".png")
            plt.close(fig)
    print("figures generated")



if __name__ == "__main__":
    args = sys.argv[1:]
    #score_files should be the final scores, sumed for all the cv... This is the new "score_tab_threeclass_cov03.pkl"
    score_files, save_path = open_score_files(args)
    score_files.reset_index(drop=True, inplace=True)
    generate_figs(score_files, save_path)
    print(args)


