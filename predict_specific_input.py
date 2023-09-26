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


def get_data(path_to_data):
    # open pickle of embedded data
    with open(path_to_data, "rb") as f:
        predict_tab = pickle.load(f)
    predict_tab.reset_index(inplace=True, drop=True)
    return predict_tab


def open_model(model_path):
    with open(model_path, "rb") as f:
        clf = pickle.load(f)
    return clf

def predict_labels(predict_tab, clf):
    y_test = predict_tab["nsub"]
    y_test = y_test.values.astype(int)
    X_test = predict_tab['esm_embeddings'].tolist()
    le = LabelEncoder()
    y_test = le.fit_transform(y_test)
    le_dict = dict(zip(le.classes_, le.transform(le.classes_)))
    print(le_dict)
    with open('le_dict_htra1.pkl', 'wb') as f:
        pickle.dump(le_dict, f)
    proba = clf.predict_proba(X_test)
    pickle.dump(proba, open("/vol/ek/Home/orlyl02/working_dir/oligopred/htra1_analysis/proba_model_2.pkl", "wb"))
    return proba


def combine_pred_actual(predict_tab, proba):
    proba_df = pd.DataFrame(proba)
    inv_map = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 10, 9: 12, 10: 14, 11: 24}
    proba_df.rename(inv_map, axis=1, inplace=True)
    pred_tab_prob = pd.concat([predict_tab, proba_df], axis=1)



if __name__ == "__main__":
    path_to_data = "/vol/ek/Home/orlyl02/working_dir/oligopred/esmfold_prediction/esm_tab_htra1.pkl"
    predict_tab = get_data(path_to_data)
    model_path = "/vol/ek/Home/orlyl02/working_dir/oligopred/mlp/esm_runs/5cv_sets_proba_models/mlp_clf_8020_downsample3_cv2.pkl"
    clf = open_model(model_path)
    proba = predict_labels(predict_tab, clf)
    combine_pred_actual(predict_tab, proba)

