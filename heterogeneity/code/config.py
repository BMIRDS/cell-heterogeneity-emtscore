import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument(
    "--traindata_folder",
    type=Path,
    default=Path("../data/traindata_csv"),
    help="input csv data directory",
)
parser.add_argument(
    "--testdata_folder",
    type=Path,
    default=Path("../data/testdata_csv"),
    help="test dataset",
)
parser.add_argument(
    "--result_folder",
    type=Path,
    default=Path("../result"),
    help="test results are saved in this directory",
)
parser.add_argument(
    "--model_path",
    type=Path,
    default=Path("../model"),
    help="trained model is saved  in this directory",
)
parser.add_argument(
    "--model_name",
    type=str,
    default="model_13_07_2021.pkl",
    help="model name which is used in testing",
)
parser.add_argument("--label_file", type=Path, default=Path("../data/label/label.csv"))
parser.add_argument(
    "--train_status", type=bool, default="True", help="retraining the model"
)
parser.add_argument(
    "--kfold", type=int, default=5, help="kfold for feature selection process"
)
parser.add_argument(
    "--split_ratio",
    type=float,
    default=0.3,
    help="split training and validation with the given ratio",
)

# these three arguments are used to convert a folder of txt files to csv files
# and from csv file to a new csv file with new columns that are only used in the analysis
parser.add_argument("--txt_folder", type=Path, default=Path("../data/testdata_txt"))
parser.add_argument(
    "--csv_folder_unfiltered",
    type=Path,
    default=Path("../data/testdata_csv_unfiltered"),
)
parser.add_argument("--csv_folder", type=Path, default=Path("../data/testdata_csv"))

args = parser.parse_args()
