# =============================================================================
# Author: Behnaz Abdollahi
# Created by: Hassanpour's Lab
# Last modification date: July 2021
# =============================================================================
# Description:
# Finds heterogeniety label using previously trained model for new images and saves the result.
# =============================================================================

import config
import csv
import joblib
from data import Data
from data import FeatureExtraction
from pathlib import Path


def test_pipeline():
    """Testing trained model on available trained model.
    Trained model and its feature transformation are found under "result" folder.
        -model_.....pkl
        -transform....pkl
    Step0: Reading the original data only columns which includes "mean" marker information.
    Step1: Combine and finds the entropy of all combination of the "mean" columns.
    Step2: Applying the saved transform on the data from step1.
    Step3: Reading the model and finds the prediction of each image.
    Step4: Saving results in "result" folder.
    """

    testdata_folder = config.args.testdata_folder
    model_name = config.args.model_name
    model_path = config.args.model_path
    result_folder = config.args.result_folder

    if not Data.check_fileformat(testdata_folder):
        print(f"Please check the file formats to be csv in {config.args.data_folder}")
        exit(-1)
    transform_name = "transform" + model_name.split("model")[1]
    if not transform_name:
        print("you need to copy the corresponding feature transform for the model")
        exit(-1)
    if not result_folder.exists():
        result_folder.mkdir(exist_ok=True)

    model = joblib.load(Path(config.args.model_path, model_name))
    transform = joblib.load(Path(model_path, transform_name))
    feature_extraction = FeatureExtraction(data_folder=testdata_folder)
    print("Transforming features similar to training...")
    combination_seven_cols = feature_extraction.combine_mean_filter_list()
    (
        X_test,
        _,
        total_fnames,
        names_pair,
    ) = feature_extraction.gen_total_fv_mean_combination(
        combination_seven_cols=combination_seven_cols,
        train_status=False,
    )

    y_pred = model.predict(transform.transform(X_test))
    print(f"Saving results...{Path(config.args.result_folder,'test_res.csv')}")
    out = open(Path(result_folder, "test_res.csv"), "w", newline="")
    writer = csv.DictWriter(out, fieldnames=["name", "y_pred"])
    writer.writeheader()
    for (y, filename) in zip(y_pred, total_fnames):
        writer.writerow({"name": filename, "y_pred": feature_extraction.map_label(y)})


if __name__ == "__main__":
    test_pipeline()
