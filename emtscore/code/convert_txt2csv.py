import config
import csv
import pandas as pd
import shutil
from pathlib import Path

def convert_txt2csv(txt_folder, csv_folder):
    # check if the txt folder exists or not empty
    if not txt_folder.exists():
        print(f"txt folder does not exists {txt_folder}")
        exit(-1) 
    if not any(txt_folder.iterdir()):
        print(f"txt folder is empty {txt_folder}")
        exit(-1)

    # delete content of the testdata_csv_unfiltered
    # create testdata_csv_unfiltered if not exists
    if not csv_folder.exists():
        csv_folder.mkdir(exist_ok=True)
    for f in csv_folder.iterdir():
        Path.unlink(f)

    # loop to read txt files and convert them to csv
    t = lambda x: True if (x.name != ".DS_Store") else False
    for fname in txt_folder.iterdir():
        if t(fname):
            txt_file = pd.read_csv(fname, delimiter="\t")
            print(
                f"converting {fname} to {str(csv_folder) + '/' + str(fname.name.split('.')[0]) + '.csv'}"
            )
            print(f"")
            txt_file.to_csv(
                str(csv_folder) + "/" + str(fname.name.split(".")[0]) + ".csv"
            )
    print("Finished!")

if __name__ == "__main__":
    txt_folder = config.args.txt_folder
    csv_folder = config.args.csv_folder_unfiltered
    convert_txt2csv(txt_folder=txt_folder, csv_folder=csv_folder)
