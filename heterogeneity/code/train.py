# =============================================================================
# Author: Behnaz Abdollahi
# Created by: Hassanpour's Lab
# Last modification date: July 2021
# =============================================================================
# Description:
# Train a new model using logistic regression on a new set of images.
# It finds the optimal features.
# =============================================================================

import collections
import config
import csv
import datetime
import joblib
import pandas as pd
from data import Data
from data import FeatureExtraction
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    accuracy_score,
    classification_report,
)
from sklearn.linear_model import LogisticRegression

def check_image_labels():
    # check that all images have the label saved in label folder
    t = lambda x: True if (x.name != ".DS_Store") else False
    check = True
    labels = pd.read_csv(config.args.label_file)
    traindata_path = config.args.traindata_folder
    label_filename, image_filename = [], []
    d = collections.defaultdict(int)
    for row in labels.itertuples():
        label_filename.append(row[1])
        d[row[1]] += 1
    for f in traindata_path.iterdir():
        if t(f):
            d[f.name.split("_cell_seg_data.csv")[0]] += 1
            image_filename.append(f.name.split("_cell_seg_data.csv")[0])

    if collections.Counter(label_filename) != collections.Counter(image_filename):
        print("Labels and image files are not identical")
        check = False
    if not check:
        for k, v in d.items():
            if v != 2:
                print(k)
    return check

def train_pipeline():
    """Training steps are run in this function.
    Step0: Reading only mean columns of all the markers
    Feature Extraction:
        Step1: Generate all possible pairs of combinination of columns from step0
        Step2: Finds the entropy of all combinations of columns
    Feature Selection:
        Step3: Finding optimal number of feature with split_ratio = 0.3
        and using RFECV python library
        Step4: Saving fearure transformation function in a .pkl file
    Step5: Fitting the regression model on transformed data
    Step6: Calculate the confusion matrix and precision, recall and F1-score
    """

    traindata_folder = config.args.traindata_folder
    labels_file = config.args.label_file
    split_ratio = config.args.split_ratio
    model_path = config.args.model_path

    if not check_image_labels():
        exit()

    if not Data.check_fileformat(traindata_folder):
        print(
            f"Please check the file formats to be csv in {traindata_folder}"
        )
        exit()
    data = Data(data_folder=config.args.traindata_folder)
    # labels = data.read_label_files()
    print("Feature extraction...")
    feature_extraction = FeatureExtraction(
        data_folder=traindata_folder, labels_file=labels_file
    )
    print("Feature selection...")
    combination_seven_cols = feature_extraction.combine_mean_filter_list()
    (
        X_orig,
        y_orig,
        total_fnames,
        names_pair,
    ) = feature_extraction.gen_total_fv_mean_combination(
        combination_seven_cols=combination_seven_cols, train_status=True
    )
    (
        X_new_train,
        X_new_test,
        y_train,
        y_test,
        indices_train,
        indices_test,
        transform,
    ) = feature_extraction.feature_selection(X_orig, y_orig, split_ratio=split_ratio)
    
    clf = LogisticRegression(random_state=14, C=1.0, class_weight='balanced')
    clf.fit(X_new_train, y_train)
    y_test_pred = clf.predict(X_new_test)
    conf_mat = confusion_matrix(y_test, y_test_pred)
    acc_ = accuracy_score(y_test, y_test_pred)
    acc = f1_score(y_test, y_test_pred, average="weighted")
    print(conf_mat)
    print(classification_report(y_test, y_test_pred))
    print("Saving trained model...")
    model_filepath = Path(
        config.args.model_path,
        "model_" + str(datetime.datetime.now().strftime("%d_%m_%Y") + ".pkl"),
    )
    joblib.dump(clf, model_filepath)
    transform_filepath = Path(
        model_path,
        "transform_" + str(datetime.datetime.now().strftime("%d_%m_%Y") + ".pkl"),
    )
    joblib.dump(transform, transform_filepath)

if __name__ == '__main__':
    train_pipeline()
