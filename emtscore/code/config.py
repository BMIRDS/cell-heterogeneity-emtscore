import argparse
from pathlib import Path

parser = argparse.ArgumentParser()

parser.add_argument(
    "--testdata_folder",
    type=Path,
    default=Path("../data/testdata_csv"),
    help="input folder should includes subfolders",
)
parser.add_argument(
    "--result_folder",
    type=Path,
    default=Path("../result"),
    help="saving scores of the images in data_folder",
)

# list of coefficients which used
parser.add_argument("--Ecadonly", type=int, default=-4)
parser.add_argument("--K8ecad", type=int, default=-3)
parser.add_argument("--K8", type=int, default=-2)
parser.add_argument("--K14", type=int, default=-2)
parser.add_argument("--Trip", type=int, default=-1)
parser.add_argument("--K8vim", type=int, default=1)
parser.add_argument("--K14vim", type=int, default=1)
parser.add_argument("--Snail", type=int, default=2)
parser.add_argument("--vimonly", type=int, default=3)
parser.add_argument("--vimzeb", type=int, default=4)

# these two arguments are used to convert a folder of txt files to csv files
parser.add_argument("--txt_folder", type=Path, default=Path("../data/testdata_txt"))
parser.add_argument("--csv_folder_unfiltered", type=Path, default=Path("../data/testdata_csv_unfiltered"))
# this argument is used to save only filtered columns and chnage their name to a standard column names that is used in code

parser.add_argument("--csv_folder", type=Path, default=Path("../data/testdata_csv"))


args = parser.parse_args()
