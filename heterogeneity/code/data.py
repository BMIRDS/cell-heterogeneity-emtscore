# =============================================================================
# Author: Behnaz Abdollahi
# Created by: Hassanpour's Lab
# Last modification date: July 2021
# =============================================================================
# Description:
# This module is used in train and test modules. It prepares inputs for training and testing.
# It read from original data, extract features, select features, apply the model.
# =============================================================================

import csv
import config
import datetime
import numpy as np
import pandas as pd
from collections import defaultdict
from itertools import chain, combinations
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import entropy
import joblib
from sklearn import preprocessing
from sklearn.feature_selection import RFECV
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression

class Data:
    """Extract features in training and test phases. Also finds optimal feature in training phase.
    Args:
        data_folder (Path): includes image files with .csv extension. If training then it should 
        be initialized with train dataset. If testing phase is running then it should be 
        initialized with test dataset.                    

    """
    def __init__(self, data_folder: Path):
        self.data_folder = data_folder

    @staticmethod
    def map_label(l):
        """Maps numbers to labels.
        Returns: 
            dictionary of labels name given their numbers

        """
        d = {1: 'low', 2: 'mid', 3:'high'}
        return d[l]

    def get_label(self, fname):
        """Reads label of the given file in label.csv file. This function is called during training
        phase.
        Args:
            fname (str): image file name
        Returns:
            label of the input image from the saved label.csv
        """
        labels = pd.read_csv(self.labels_file)
        map_label = {'low': 1, 'mid': 2, 'high': 3}
        for row in labels.itertuples():
            if row[1] == str(fname.split("_cell_seg_data.csv")[0]):
                label = map_label[row[2]]
        return label
    
    @staticmethod
    def check_fileformat(data_folder: Path):
        # Checking the format of the input files. They should be .csv files
        check = True
        t = lambda x: True if (x.name != ".DS_Store") else False
        for f in data_folder.iterdir():
            if t(f) and not f.match('*.csv'):
                print(f"file {f.name} is not csv file")
                check = False
        return check

class FeatureExtraction(Data):
    def __init__(self, data_folder, labels_file=None):
        super().__init__(data_folder)
        self.labels_file = labels_file

    def featurevector_test(self):
        """This finds one of the list of feature vector of test
        """
        fv_total, filename_list = [], []
        for fname in self.data_folder.iterdir():
            fv_total.append(self.feature_extraction_test(fname))
            filename_list.append(fname.name)
        return fv_total, filename_list

    def feature_extraction_test(self, fname):
        """
        Thirteen features which are combination of columns should be extracted from the csv file
        combined_dict save the known features name which each key represent the combination of one or more than one columns
        raw information of the columns are extracted using function entropy_tumor_mean_test
        if there are more than one column that should be combined then entropy of their combination is calculated and saved as one of the features
        Each file should be summarized into 13 feature vector
        """
        nbins = 200
        # reading the list name
        # matcing the name with the original and extract features from columns
        t = lambda x: True if (x.name != ".DS_Store") else False
        combined_dict = [
            "Cytoplasm_690, Cytoplasm_520",
            "Nucleus_570, Cytoplasm_690",
            "Nucleus_620, Cytoplasm_690, Cytoplasm_540, Cytoplasm_520",
            "EC_690, EC_650, EC_620, EC_570,EC_540,EC_520",
            "Nucleus_570",
            "Nucleus_620, Cytoplasm_690, Cytoplasm_650, Cytoplasm_520",
            "Nucleus_620, Nucleus_570, Cytoplasm_520",
            "EC_520",
            "EC_690",
            "Nucleus_620, Cytoplasm_650, Membrane_650",
            "Nucleus_620, Nucleus_570, Cytoplasm_650, Membrane_650",
            "Cytoplasm_690",
            "EC_570",
        ]
        fv_singlefile = []
        fv = defaultdict(list)
        # for fname in self.data_folder.iterdir():
        if t(fname):
            fv_cell = self.entropy_tumor_mean_test(fname)
            # reading columns names from combined_dict
            for i, v in enumerate(combined_dict):
                for cols in v.split(","):
                    fv[i].extend(fv_cell[cols])
                    # print(cols, len(fv[i]))
                hist, _ = np.histogram(fv[i], bins=nbins, density=True)
                hist = [i for i in hist if i != 0]
                fv_singlefile.append(entropy(hist, base=2))
            print(len(fv_singlefile))
        return fv_singlefile

    def entropy_tumor_mean_test(self, file_path):
        """The function finds the corresponding columns for each new patients file.
        Args:
            file_path (Path): the path of the input image

        """
        nbins = 200
        fv = defaultdict(list)
        fv_marker = defaultdict(list)
        fv_cell = defaultdict(list)
        # tumor_cells = filter_tumor(pd.read_csv(file_path))
        tumor_cells = pd.read_csv(file_path)

        entropy_marker = defaultdict(list)
        entropy_cell_marker = defaultdict(list)
        #######################################CELLL TYPE#########################################################
        #######################################Nuclues############################################################
        fv_cell["Nucleus_690"] = tumor_cells[
            "Nucleus Vim (Opal 690) Mean (Normalized Counts, Total Weighting)"
        ].dropna()
        hist, _ = np.histogram(fv_cell["Nucleus_690"], bins=nbins, density=True)
        hist = [i for i in hist if i != 0]
        entropy_cell_marker["Nucleus_690"] = entropy(hist, base=2)

        fv_cell["Nucleus_650"] = tumor_cells[
            "Nucleus Ecad (Opal 650) Mean (Normalized Counts, Total Weighting)"
        ].dropna()
        hist, _ = np.histogram(fv_cell["Nucleus_650"], bins=nbins, density=True)
        hist = [i for i in hist if i != 0]
        entropy_cell_marker["Nucleus_650"] = entropy(hist, base=2)

        fv_cell["Nucleus_620"] = tumor_cells[
            "Nucleus Snail (Opal 620) Mean (Normalized Counts, Total Weighting)"
        ].dropna()
        hist, _ = np.histogram(fv_cell["Nucleus_620"], bins=nbins, density=True)
        hist = [i for i in hist if i != 0]
        entropy_cell_marker["Nucleus_620"] = entropy(hist, base=2)
        fv_cell["Nucleus_570"] = tumor_cells[
            "Nucleus ZEB1 (Opal 570) Mean (Normalized Counts, Total Weighting)"
        ].dropna()
        hist, _ = np.histogram(fv_cell["Nucleus_570"], bins=nbins, density=True)
        hist = [i for i in hist if i != 0]
        entropy_cell_marker["Nucleus_570"] = entropy(hist, base=2)

        fv_cell["Nucleus_540"] = tumor_cells[
            "Nucleus K8 (Opal 540) Mean (Normalized Counts, Total Weighting)"
        ].dropna()
        hist, _ = np.histogram(fv_cell["Nucleus_540"], bins=nbins, density=True)
        hist = [i for i in hist if i != 0]
        entropy_cell_marker["Nucleus_540"] = entropy(hist, base=2)

        fv_cell["Nucleus_520"] = tumor_cells[
            "Nucleus K14 (Opal 520) Mean (Normalized Counts, Total Weighting)"
        ].dropna()
        hist, _ = np.histogram(fv_cell["Nucleus_520"], bins=nbins, density=True)
        hist = [i for i in hist if i != 0]
        entropy_cell_marker["Nucleus_520"] = entropy(hist, base=2)
        #######################################Entire cell############################################################
        fv_cell["EC_690"] = tumor_cells[
            "Entire Cell Vim (Opal 690) Mean (Normalized Counts, Total Weighting)"
        ].dropna()
        hist, _ = np.histogram(fv_cell["EC_690"], bins=nbins, density=True)
        hist = [i for i in hist if i != 0]
        entropy_cell_marker["EC_690"] = entropy(hist, base=2)

        fv_cell["EC_650"] = tumor_cells[
            "Entire Cell Ecad (Opal 650) Mean (Normalized Counts, Total Weighting)"
        ].dropna()
        hist, _ = np.histogram(fv_cell["EC_650"], bins=nbins, density=True)
        hist = [i for i in hist if i != 0]
        entropy_cell_marker["EC_650"] = entropy(hist, base=2)

        fv_cell["EC_620"] = tumor_cells[
            "Entire Cell Snail (Opal 620) Mean (Normalized Counts, Total Weighting)"
        ].dropna()
        hist, _ = np.histogram(fv_cell["EC_620"], bins=nbins, density=True)
        hist = [i for i in hist if i != 0]
        entropy_cell_marker["EC_620"] = entropy(hist, base=2)

        fv_cell["EC_570"] = tumor_cells[
            "Entire Cell ZEB1 (Opal 570) Mean (Normalized Counts, Total Weighting)"
        ].dropna()
        hist, _ = np.histogram(fv_cell["EC_570"], bins=nbins, density=True)
        hist = [i for i in hist if i != 0]
        entropy_cell_marker["EC_570"] = entropy(hist, base=2)

        fv_cell["EC_540"] = tumor_cells[
            "Entire Cell K8 (Opal 540) Mean (Normalized Counts, Total Weighting)"
        ].dropna()
        hist, _ = np.histogram(fv_cell["EC_540"], bins=nbins, density=True)
        hist = [i for i in hist if i != 0]
        entropy_cell_marker["EC_540"] = entropy(hist, base=2)

        fv_cell["EC_520"] = tumor_cells[
            "Entire Cell K14 (Opal 520) Mean (Normalized Counts, Total Weighting)"
        ].dropna()
        hist, _ = np.histogram(fv_cell["EC_520"], bins=nbins, density=True)
        hist = [i for i in hist if i != 0]
        entropy_cell_marker["EC_520"] = entropy(hist, base=2)

        #######################################Cytoplasm############################################################
        fv_cell["Cytoplasm_690"] = tumor_cells[
            "Cytoplasm Vim (Opal 690) Mean (Normalized Counts, Total Weighting)"
        ].dropna()
        hist, _ = np.histogram(fv_cell["Cytoplasm_690"], bins=nbins, density=True)
        hist = [i for i in hist if i != 0]
        entropy_cell_marker["Cytoplasm_690"] = entropy(hist, base=2)

        fv_cell["Cytoplasm_650"] = tumor_cells[
            "Cytoplasm Ecad (Opal 650) Mean (Normalized Counts, Total Weighting)"
        ].dropna()
        hist, _ = np.histogram(fv_cell["Cytoplasm_650"], bins=nbins, density=True)
        hist = [i for i in hist if i != 0]
        entropy_cell_marker["Cytoplasm_650"] = entropy(hist, base=2)

        fv_cell["Cytoplasm_620"] = tumor_cells[
            "Cytoplasm Snail (Opal 620) Mean (Normalized Counts, Total Weighting)"
            ].dropna()
        hist, _ = np.histogram(fv_cell["Cytoplasm_620"], bins=nbins, density=True)
        hist = [i for i in hist if i != 0]
        entropy_cell_marker["Cytoplasm_620"] = entropy(hist, base=2)

        fv_cell["Cytoplasm_570"] = tumor_cells[
            "Cytoplasm ZEB1 (Opal 570) Mean (Normalized Counts, Total Weighting)"
        ].dropna()
        hist, _ = np.histogram(fv_cell["Cytoplasm_570"], bins=nbins, density=True)
        hist = [i for i in hist if i != 0]
        entropy_cell_marker["Cytoplasm_570"] = entropy(hist, base=2)

        fv_cell["Cytoplasm_540"] = tumor_cells[
            "Cytoplasm K8 (Opal 540) Mean (Normalized Counts, Total Weighting)"
        ].dropna()
        hist, _ = np.histogram(fv_cell["EC_540"], bins=nbins, density=True)
        hist = [i for i in hist if i != 0]
        entropy_cell_marker["Cytoplasm_540"] = entropy(hist, base=2)

        fv_cell["Cytoplasm_520"] = tumor_cells[
            "Cytoplasm K14 (Opal 520) Mean (Normalized Counts, Total Weighting)"
        ].dropna()
        hist, _ = np.histogram(fv_cell["Cytoplasm_520"], bins=nbins, density=True)
        hist = [i for i in hist if i != 0]
        entropy_cell_marker["Cytoplasm_520"] = entropy(hist, base=2)
        #######################################Membrane############################################################
        fv_cell["Membrane_690"] = tumor_cells[
            "Membrane Vim (Opal 690) Mean (Normalized Counts, Total Weighting)"
        ].dropna()
        hist, _ = np.histogram(fv_cell["Membrane_690"], bins=nbins, density=True)
        hist = [i for i in hist if i != 0]
        entropy_cell_marker["Membrane_690"] = entropy(hist, base=2)

        fv_cell["Membrane_650"] = tumor_cells[
            "Membrane Ecad (Opal 650) Mean (Normalized Counts, Total Weighting)"
        ].dropna()
        hist, _ = np.histogram(fv_cell["Membrane_650"], bins=nbins, density=True)
        hist = [i for i in hist if i != 0]
        entropy_cell_marker["Membrane_650"] = entropy(hist, base=2)

        fv_cell["Membrane_620"] = tumor_cells[
            "Membrane Snail (Opal 620) Mean (Normalized Counts, Total Weighting)"
        ].dropna()
        hist, _ = np.histogram(fv_cell["Membrane_620"], bins=nbins, density=True)
        hist = [i for i in hist if i != 0]
        entropy_cell_marker["Membrane_620"] = entropy(hist, base=2)

        fv_cell["Membrane_570"] = tumor_cells[
            "Membrane ZEB1 (Opal 570) Mean (Normalized Counts, Total Weighting)"
        ].dropna()
        hist, _ = np.histogram(fv_cell["Cytoplasm_570"], bins=nbins, density=True)
        hist = [i for i in hist if i != 0]
        entropy_cell_marker["Membrane_570"] = entropy(hist, base=2)

        fv_cell["Membrane_540"] = tumor_cells[
            "Membrane K8 (Opal 540) Mean (Normalized Counts, Total Weighting)"
        ].dropna()
        hist, _ = np.histogram(fv_cell["EC_540"], bins=nbins, density=True)
        hist = [i for i in hist if i != 0]
        entropy_cell_marker["Membrane_540"] = entropy(hist, base=2)

        fv_cell["Membrane_520"] = tumor_cells[
            "Membrane K14 (Opal 520) Mean (Normalized Counts, Total Weighting)"
        ].dropna()
        hist, _ = np.histogram(fv_cell["Membrane_520"], bins=nbins, density=True)
        hist = [i for i in hist if i != 0]
        entropy_cell_marker["Membrane_520"] = entropy(hist, base=2)
        # print("entropy_cell_marker",entropy_cell_marker)

        ########################################################
        #     fv_marker['520'].extend(tumor_cells['Entire Cell Opal 520 Mean (Normalized Counts, Total Weighting)'].dropna())
        #     hist, _ = np.histogram(fv_marker['520'], bins=nbins, density=True)
        #     hist = [i for i in hist if i != 0]

        #     fv_marker['520'].extend(tumor_cells['Nucleus Opal 520 Mean (Normalized Counts, Total Weighting)'].dropna())
        #     hist, _ = np.histogram(fv_marker['520'], bins=nbins, density=True)
        #     hist = [i for i in hist if i != 0]

        #     fv_marker['520'].extend(tumor_cells['Membrane Opal 520 Mean (Normalized Counts, Total Weighting)'].dropna())
        #     hist, _ = np.histogram(fv_marker['520'], bins=nbins, density=True)
        #     hist = [i for i in hist if i != 0]

        #     fv_marker['520'].extend(tumor_cells['Cytoplasm Opal 520 Mean (Normalized Counts, Total Weighting)'].dropna())
        #     hist, _ = np.histogram(fv_marker['520'], bins=nbins, density=True)
        #     hist = [i for i in hist if i != 0]

        #     entropy_marker['520']=entropy(hist, base=2)
        return fv_cell

    @staticmethod
    def combine_mean_filter_list():
        # All columns should be combined each of them which generates all the possible combinations
        # of columns
        all_combination = []
        fv_inputs_nums = [0, 1, 2, 3, 4, 5, 6]
        rs = [1, 2, 3, 4, 5, 6, 7]
        for r in rs:
            # print(list(combinations(fv_inuts_nums, 2)))
            all_combination.extend(list(combinations(fv_inputs_nums, r)))
        return all_combination

    @staticmethod
    def filter_tumor(data_):
        cols = data_.columns
        if 'Tissue Category' in cols:
            return data_[data_["Tissue Category"] == "tumor"]
        else:
            return data_

    def gen_total_fv_mean_combination(self, combination_seven_cols, train_status):
        t = lambda x: True if (x.name != ".DS_Store") else False
        total_fv, total_label, total_fnames = [], [], []
        for fname in self.data_folder.iterdir():
            if t(fname):
                _, fv_list = self.entropy_tumor_mean_filter_list_before_combine(fname)
                fv, names_pair = self.entropy_tumor_mean_filter_list_after_combine(
                    combination_seven_cols, fv_list[0]
                )
                # fv = self.total_tumor_mean_filter_list_after_combine(combination_seven_cols, fv_list[0])
                total_fv.append(fv)
                # print(fname, "len(fv)",len(fv))
                if train_status:
                    total_label.append(self.get_label(fname.name))
                total_fnames.append(((fname.name).split("_cell_seg_data.csv")[0]))
        return total_fv, total_label, total_fnames, names_pair


    def entropy_tumor_mean_filter_list_before_combine(self, file_path):
        """Finds the entropy of the columns before combining them
        """
        # min_max_scaler = preprocessing.MinMaxScaler()
        fv_cell = defaultdict(list)
        tumor_cells = self.filter_tumor(pd.read_csv(file_path))
        entropy_cell_marker = defaultdict(list)
        fv_list = []

        ######Nuclear 620,Nuclear 570,Cytoplasm 690,Cytoplasm 650,Cytoplasm 540,Cytoplasm 520,Membrane 650######
        fv_cell["Nucleus_620"] = tumor_cells[
            "Nucleus Opal 620 Mean (Normalized Counts, Total Weighting)"
        ].dropna()
        fv_cell["Nucleus_620"] = preprocessing.minmax_scale(
            fv_cell["Nucleus_620"], feature_range=(0, 1)
        )
        hist, _ = np.histogram(fv_cell["Nucleus_620"], bins=50, density=True)
        hist = [i for i in hist if i != 0]
        entropy_cell_marker["Nucleus_620"] = entropy(hist, base=2)

        fv_cell["Nucleus_570"] = tumor_cells[
            "Nucleus Opal 570 Mean (Normalized Counts, Total Weighting)"
        ].dropna()
        fv_cell["Nucleus_570"] = preprocessing.minmax_scale(
            fv_cell["Nucleus_570"], feature_range=(0, 1)
        )
        hist, _ = np.histogram(fv_cell["Nucleus_570"], bins=50, density=True)
        hist = [i for i in hist if i != 0]
        entropy_cell_marker["Nucleus_570"] = entropy(hist, base=2)

        fv_cell["Cytoplasm_690"] = tumor_cells[
            "Cytoplasm Opal 690 Mean (Normalized Counts, Total Weighting)"
        ].dropna()
        fv_cell["Cytoplasm_690"] = preprocessing.minmax_scale(
            fv_cell["Cytoplasm_690"], feature_range=(0, 1)
        )
        hist, _ = np.histogram(fv_cell["Cytoplasm_690"], bins=50, density=True)
        hist = [i for i in hist if i != 0]
        entropy_cell_marker["Cytoplasm_690"] = entropy(hist, base=2)

        fv_cell["Cytoplasm_650"] = tumor_cells[
            "Cytoplasm Opal 650 Mean (Normalized Counts, Total Weighting)"
        ].dropna()
        fv_cell["Cytoplasm_650"] = preprocessing.minmax_scale(
            fv_cell["Cytoplasm_650"], feature_range=(0, 1)
        )
        hist, _ = np.histogram(fv_cell["Cytoplasm_650"], bins=50, density=True)
        hist = [i for i in hist if i != 0]
        entropy_cell_marker["Cytoplasm_650"] = entropy(hist, base=2)

        fv_cell["Cytoplasm_540"] = tumor_cells[
            "Cytoplasm Opal 540 Mean (Normalized Counts, Total Weighting)"
        ].dropna()
        fv_cell["Cytoplasm_540"] = preprocessing.minmax_scale(
            fv_cell["Cytoplasm_540"], feature_range=(0, 1)
        )
        hist, _ = np.histogram(fv_cell["Cytoplasm_540"], bins=50, density=True)
        hist = [i for i in hist if i != 0]
        entropy_cell_marker["Cytoplasm_540"] = entropy(hist, base=2)

        fv_cell["Cytoplasm_520"] = tumor_cells[
            "Cytoplasm Opal 520 Mean (Normalized Counts, Total Weighting)"
        ].dropna()
        fv_cell["Cytoplasm_520"] = preprocessing.minmax_scale(
            fv_cell["Cytoplasm_520"], feature_range=(0, 1)
        )
        hist, _ = np.histogram(fv_cell["Cytoplasm_520"], bins=50, density=True)
        hist = [i for i in hist if i != 0]
        entropy_cell_marker["Cytoplasm_520"] = entropy(hist, base=2)

        fv_cell["Membrane_650"] = tumor_cells[
            "Membrane Opal 650 Mean (Normalized Counts, Total Weighting)"
        ].dropna()
        fv_cell["Membrane_650"] = preprocessing.minmax_scale(
            fv_cell["Membrane_650"], feature_range=(0, 1)
        )
        hist, _ = np.histogram(fv_cell["Membrane_650"], bins=50, density=True)
        hist = [i for i in hist if i != 0]
        entropy_cell_marker["Membrane_650"] = entropy(hist, base=2)

        #######################################Entire cell############################################################
        fv_cell["EC_690"] = tumor_cells[
            "Entire Cell Opal 690 Mean (Normalized Counts, Total Weighting)"
        ].dropna()
        fv_cell["EC_690"] = preprocessing.minmax_scale(
            fv_cell["EC_690"], feature_range=(0, 1)
        )
        hist, _ = np.histogram(fv_cell["EC_690"], bins=50, density=True)
        hist = [i for i in hist if i != 0]
        entropy_cell_marker["EC_690"] = entropy(hist, base=2)

        fv_cell["EC_650"] = tumor_cells[
            "Entire Cell Opal 650 Mean (Normalized Counts, Total Weighting)"
        ].dropna()
        fv_cell["EC_650"] = preprocessing.minmax_scale(
            fv_cell["EC_650"], feature_range=(0, 1)
        )
        hist, _ = np.histogram(fv_cell["EC_650"], bins=50, density=True)
        hist = [i for i in hist if i != 0]
        entropy_cell_marker["EC_650"] = entropy(hist, base=2)

        fv_cell["EC_620"] = tumor_cells[
            "Entire Cell Opal 620 Mean (Normalized Counts, Total Weighting)"
            ].dropna()
        fv_cell["EC_620"] = preprocessing.minmax_scale(
            fv_cell["EC_620"], feature_range=(0, 1)
        )
        hist, _ = np.histogram(fv_cell["EC_620"], bins=50, density=True)
        hist = [i for i in hist if i != 0]
        entropy_cell_marker["EC_620"] = entropy(hist, base=2)

        fv_cell["EC_570"] = tumor_cells[
            "Entire Cell Opal 570 Mean (Normalized Counts, Total Weighting)"
        ].dropna()
        fv_cell["EC_570"] = preprocessing.minmax_scale(
            fv_cell["EC_570"], feature_range=(0, 1)
        )
        hist, _ = np.histogram(fv_cell["EC_570"], bins=50, density=True)
        hist = [i for i in hist if i != 0]
        entropy_cell_marker["EC_570"] = entropy(hist, base=2)

        fv_cell["EC_540"] = tumor_cells[
            "Entire Cell Opal 540 Mean (Normalized Counts, Total Weighting)"
        ].dropna()
        fv_cell["EC_540"] = preprocessing.minmax_scale(
            fv_cell["EC_540"], feature_range=(0, 1)
        )
        hist, _ = np.histogram(fv_cell["EC_540"], bins=50, density=True)
        hist = [i for i in hist if i != 0]
        entropy_cell_marker["EC_540"] = entropy(hist, base=2)

        fv_cell["EC_520"] = tumor_cells[
            "Entire Cell Opal 520 Mean (Normalized Counts, Total Weighting)"
        ].dropna()
        fv_cell["EC_520"] = preprocessing.minmax_scale(
            fv_cell["EC_520"], feature_range=(0, 1)
        )
        hist, _ = np.histogram(fv_cell["EC_520"], bins=50, density=True)
        hist = [i for i in hist if i != 0]
        entropy_cell_marker["EC_520"] = entropy(hist, base=2)

        fv_list.append(
            [
                fv_cell["Nucleus_620"],
                fv_cell["Nucleus_570"],
                fv_cell["Cytoplasm_690"],
                fv_cell["Cytoplasm_650"],
                fv_cell["Cytoplasm_540"],
                fv_cell["Cytoplasm_520"],
                fv_cell["Membrane_650"],
                fv_cell["EC_690"],
                fv_cell["EC_650"],
                fv_cell["EC_620"],
                fv_cell["EC_570"],
                fv_cell["EC_540"],
                fv_cell["EC_520"],
            ]
        )
        return entropy_cell_marker, fv_list

    def entropy_tumor_mean_filter_list_after_combine(
        self, combination_seven_cols, fv_list_mean
    ):
        """Finds all the possible pairs of combinations of all mean columns.
        """
        # The assumption is that the first seven columns of fv_cell_mean belong to the 7 columns
        # Last six columns belong to entire cell
        nbins = 200
        entropy_pairs = []
        fv_pairs = []
        names_pair = []
        for rc in combination_seven_cols:
            fv_pairs = []
            if len(rc) == 1:
                fv_pairs.extend(fv_list_mean[rc[0]])
                hist, _ = np.histogram(fv_pairs, bins=nbins, density=True)
                hist = [i for i in hist if i != 0]
                entropy_pairs.append(entropy(hist, base=2))
                names_pair.append([rc[0]])
            fv_pairs = []
            if len(rc) == 2:
                # print(len(fv_list_mean[rc[0]]), len(fv_list_mean[rc[1]]))
                fv_pairs.extend(fv_list_mean[rc[0]])
                fv_pairs.extend(fv_list_mean[rc[1]])
                hist, _ = np.histogram(fv_pairs, bins=nbins, density=True)
                hist = [i for i in hist if i != 0]
                entropy_pairs.append(entropy(hist, base=2))
                names_pair.append([rc[0], rc[1]])
            fv_pairs = []
            if len(rc) == 3:
                fv_pairs.extend(fv_list_mean[rc[0]])
                fv_pairs.extend(fv_list_mean[rc[1]])
                fv_pairs.extend(fv_list_mean[rc[2]])
                hist, _ = np.histogram(fv_pairs, bins=nbins, density=True)
                hist = [i for i in hist if i != 0]
                entropy_pairs.append(entropy(hist, base=2))
                names_pair.append([rc[0], rc[1], rc[2]])
            fv_pairs = []
            if len(rc) == 4:
                fv_pairs.extend(fv_list_mean[rc[0]])
                fv_pairs.extend(fv_list_mean[rc[1]])
                fv_pairs.extend(fv_list_mean[rc[2]])
                fv_pairs.extend(fv_list_mean[rc[3]])
                hist, _ = np.histogram(fv_pairs, bins=nbins, density=True)
                hist = [i for i in hist if i != 0]
                entropy_pairs.append(entropy(hist, base=2))
                names_pair.append([rc[0], rc[1], rc[2], rc[3]])
            fv_pairs = []
            if len(rc) == 5:
                fv_pairs.extend(fv_list_mean[rc[0]])
                fv_pairs.extend(fv_list_mean[rc[1]])
                fv_pairs.extend(fv_list_mean[rc[2]])
                fv_pairs.extend(fv_list_mean[rc[3]])
                fv_pairs.extend(fv_list_mean[rc[4]])
                hist, _ = np.histogram(fv_pairs, bins=nbins, density=True)
                hist = [i for i in hist if i != 0]
                entropy_pairs.append(entropy(hist, base=2))
                names_pair.append([rc[0], rc[1], rc[2], rc[3], rc[4]])
            fv_pairs = []
            if len(rc) == 6:
                fv_pairs.extend(fv_list_mean[rc[0]])
                fv_pairs.extend(fv_list_mean[rc[1]])
                fv_pairs.extend(fv_list_mean[rc[2]])
                fv_pairs.extend(fv_list_mean[rc[3]])
                fv_pairs.extend(fv_list_mean[rc[4]])
                fv_pairs.extend(fv_list_mean[rc[5]])
                hist, _ = np.histogram(fv_pairs, bins=nbins, density=True)
                hist = [i for i in hist if i != 0]
                entropy_pairs.append(entropy(hist, base=2))
                names_pair.append([rc[0], rc[1], rc[2], rc[3], rc[4], rc[5]])
            fv_pairs = []
            if len(rc) == 7:
                fv_pairs.extend(fv_list_mean[rc[0]])
                fv_pairs.extend(fv_list_mean[rc[1]])
                fv_pairs.extend(fv_list_mean[rc[2]])
                fv_pairs.extend(fv_list_mean[rc[3]])
                fv_pairs.extend(fv_list_mean[rc[4]])
                fv_pairs.extend(fv_list_mean[rc[5]])
                fv_pairs.extend(fv_list_mean[rc[6]])
                hist, _ = np.histogram(fv_pairs, bins=nbins, density=True)
                hist = [i for i in hist if i != 0]
                entropy_pairs.append(entropy(hist, base=2))
                names_pair.append([rc[0], rc[1], rc[2], rc[3], rc[4], rc[5], rc[6]])
        
        # Here we append the entire cell features entropies
        fv_pairs = []
        for i in range(7, 13):
            fv_single = []
            fv_single.extend(fv_list_mean[i])
            fv_pairs.extend(fv_list_mean[i])
            hist, _ = np.histogram(fv_single, bins=nbins, density=True)
            hist = [i for i in hist if i != 0]
            entropy_pairs.append(entropy(hist, base=2))
            names_pair.append([i])
        hist, _ = np.histogram(fv_pairs, bins=nbins, density=True)
        hist = [i for i in hist if i != 0]
        entropy_pairs.append(entropy(hist, base=2))
        names_pair.append(["end"])
        return entropy_pairs, names_pair

    def feature_selection(self, X, y, split_ratio=config.args.split_ratio):
        """Finds the optimal number of features from all the feature vectors

        """
        rfecv = RFECV(
            estimator=LogisticRegression(random_state=14,class_weight='balanced',max_iter=10000,C=1.0),
            # estimator=LogisticRegression(random_state=14,max_iter=10000),
            step=1,
            cv=StratifiedKFold(config.args.kfold),
            scoring="accuracy",
        )
        print(f"Fitting with Kfold = {config.args.kfold}")
        print("It may take longe time if k is larger")
        rfecv.fit(X, y)

        indices = np.arange(len(X))
        (
            X_new_train,
            X_new_test,
            y_train,
            y_test,
            indices_train,
            indices_test,
        ) = train_test_split(
                rfecv.transform(X), y, indices, test_size=split_ratio, random_state=0
                )
        print("Optimal number of features : %d" % rfecv.n_features_)
        plt.figure()
        plt.xlabel("Number of features selected")
        plt.ylabel("Cross validation score (nb of correct classifications)")
        plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
        plt.show()
        return X_new_train, X_new_test, y_train, y_test, indices_train, indices_test, rfecv

