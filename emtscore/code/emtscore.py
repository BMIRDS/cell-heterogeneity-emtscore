# =============================================================================
# Author: Behnaz Abdollahi
# Created by: Hassanpour's Lab
# Final modification date: July 2021
# =============================================================================
# Description:
# This module finds emtscore using predefined coefficients.
# =============================================================================

import config
import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from pathlib import Path


class EMTScore:
    """Finds emtscore of new images using known coefficients.
    Args:
        folder_data (Path): input images to find their scores.

    """
    def __init__(self, folder_data):
        self.folder_data = folder_data

    @staticmethod
    def sigmoid(z):
        # Calculate the sigmoid of given number.
        return 1 / (1 + np.exp(-z))

    def filter_tumor(self, data_):
        # Finds if 'Tissue Category' columns is in the given data columns 
        if "Tissue Category" not in data_.columns:
            return data_
        else:
            return data_[data_["Tissue Category"] == "tumor"]

    def read_data_phenotype(self, file_path):
        # Reads "Phenotype" columns of a singel image file.
        tumor_cells = self.filter_tumor(pd.read_csv(file_path))
        tumor_cells = tumor_cells.dropna(subset=["Phenotype"])
        phenotype = tumor_cells["Phenotype"]
        return phenotype

    def gen_results(self, filename, pheno):
        # Reads image file and returns the score of each file in a dictionary 
        # key of the dictionary is the filename and value is its score
        phenotype_count_dict, phenotype_prob_dict = self.count_phenotype(
            phenotype_fv=pheno, filename=filename
        )
        return self.score_sigmoid(phenotype_prob_dict)


    def count_phenotype(self, filename, phenotype_fv):
        # Counts number of each phenotype and finds their ratios.
        # Eight phenotypes are defined.
        new_s = {}
        new_s["filename"] = filename
        new_s["vim only"] = 0
        new_s["Ecad only"] = 0
        new_s["K8+ecad"] = 0
        new_s["K14"] = 0
        new_s["K8"] = 0
        new_s["Trip+"] = 0
        new_s["K8+vim"] = 0
        new_s["Snail+"] = 0
        new_s["vim+zeb"] = 0
        new_s["total"] = 0
        prob_dict = defaultdict(int)
        s = dict(Counter(phenotype_fv))
        for k, v in s.items():
            if k == "K8" or k == "K14" or k == "k14":
                new_s["K14"] += v
            else:
                new_s[k] += v
        new_s["total"] += (
            new_s["vim only"]
            + new_s["Ecad only"]
            + new_s["K8+ecad"]
            + new_s["K14"]
            + new_s["K8"]
            + new_s["Trip+"]
            + new_s["K8+vim"]
            + new_s["Snail+"]
            + new_s["vim+zeb"]
        )
        for k, v in new_s.items():
            if k != "filename":
                prob_dict[k] = float("{:.2f}".format(v / (new_s["total"])))
        return new_s, prob_dict


    def score_sigmoid(self, phenotype_prob_dict_curr):
        # Reads the phenotype_column data of the given image
        # Calculates the score of the phenotypes using known coefficients.

        curr = 0
        coefficients = {
            "Ecad only": config.args.Ecadonly,
            "K8+ecad": config.args.K8ecad,
            "K8": config.args.K8,
            "K14": config.args.K14,
            "Snail+": config.args.Snail,
            "Trip+": config.args.Trip,
            "K8+vim": config.args.K8vim,
            "vim only": config.args.vimonly,
            "vim+zeb": config.args.vimzeb,
        }
        for k, v in phenotype_prob_dict_curr.items():
            if k in coefficients.keys():
                curr += coefficients[k] * phenotype_prob_dict_curr[k]
        return self.sigmoid(curr)


def find_score_folder():
    # Read images from the testdata_folder
    # Calculate score of each images
    # Save them in folder result
    testdata_folder = config.args.testdata_folder
    result_folder = config.args.result_folder
    if not testdata_folder.exists():
        print(f"Directory is empty {testdata_folder}")
    if not result_folder.exists():
        result_folder.mkdir(exist_ok=True)

    t = lambda x: True if (x.name != ".DS_Store") else False
    scores = defaultdict()
    emtscore = EMTScore(folder_data=testdata_folder)
    for f in testdata_folder.iterdir():
        print(f"Calculating {f.name}")
        if t(f):
            pheno_info = emtscore.read_data_phenotype(f)
            scores[f.name] = emtscore.gen_results(f.name, pheno_info)
    fres = open(Path(result_folder, "scores.csv"), "w", newline="")
    writer = csv.DictWriter(fres, fieldnames=["name", "score"])
    writer.writeheader()
    for k, v in scores.items():
        writer.writerow({"name": k, "score": f"{v:.4f}"})

if __name__ == "__main__":
    find_score_folder()
