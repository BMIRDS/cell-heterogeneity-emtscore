import collections
import config
import csv
import pandas as pd
import re
from pathlib import Path


def convert_colnames(src, des):

    standard_cols = collections.defaultdict(list)
    target_colnames = [
        "Nucleus Opal 690 Mean (Normalized Counts, Total Weighting)",
        "Nucleus Opal 650 Mean (Normalized Counts, Total Weighting)",
        "Nucleus Opal 620 Mean (Normalized Counts, Total Weighting)",
        "Nucleus Opal 570 Mean (Normalized Counts, Total Weighting)",
        "Nucleus Opal 540 Mean (Normalized Counts, Total Weighting)",
        "Nucleus Opal 520 Mean (Normalized Counts, Total Weighting)",
        "Entire Cell Opal 690 Mean (Normalized Counts, Total Weighting)",
        "Entire Cell Opal 650 Mean (Normalized Counts, Total Weighting)",
        "Entire Cell Opal 620 Mean (Normalized Counts, Total Weighting)",
        "Entire Cell Opal 570 Mean (Normalized Counts, Total Weighting)",
        "Entire Cell Opal 540 Mean (Normalized Counts, Total Weighting)",
        "Entire Cell Opal 520 Mean (Normalized Counts, Total Weighting)",
        "Cytoplasm Opal 690 Mean (Normalized Counts, Total Weighting)",
        "Cytoplasm Opal 650 Mean (Normalized Counts, Total Weighting)",
        "Cytoplasm Opal 620 Mean (Normalized Counts, Total Weighting)",
        "Cytoplasm Opal 570 Mean (Normalized Counts, Total Weighting)",
        "Cytoplasm Opal 540 Mean (Normalized Counts, Total Weighting)",
        "Cytoplasm Opal 520 Mean (Normalized Counts, Total Weighting)",
        "Membrane Opal 690 Mean (Normalized Counts, Total Weighting)",
        "Membrane Opal 650 Mean (Normalized Counts, Total Weighting)",
        "Membrane Opal 620 Mean (Normalized Counts, Total Weighting)",
        "Membrane Opal 570 Mean (Normalized Counts, Total Weighting)",
        "Membrane Opal 540 Mean (Normalized Counts, Total Weighting)",
        "Membrane Opal 520 Mean (Normalized Counts, Total Weighting)",
    ]
    
    f = open(src, "r")
    out = open(Path(des, src.name), "w", newline="")

    reader = csv.DictReader(f)
    headers = reader.fieldnames

    keywords = ["Nucleus", "Mean", "Vim"]
    patterns = {
        "m1": ["Nucleus", "Mean", "690"],
        "m2": ["Nucleus", "Mean", "650"],
        "m3": ["Nucleus", "Mean", "620"],
        "m4": ["Nucleus", "Mean", "570"],
        "m5": ["Nucleus", "Mean", "540"],
        "m6": ["Nucleus", "Mean", "520"],
        "m7": ["Entire", "Mean", "690"],
        "m8": ["Entire", "Mean", "650"],
        "m9": ["Entire", "Mean", "620"],
        "m10": ["Entire", "Mean", "570"],
        "m11": ["Entire", "Mean", "540"],
        "m12": ["Entire", "Mean", "520"],
        "m13": ["Cytoplasm", "Mean", "690"],
        "m14": ["Cytoplasm", "Mean", "650"],
        "m15": ["Cytoplasm", "Mean", "620"],
        "m16": ["Cytoplasm", "Mean", "570"],
        "m17": ["Cytoplasm", "Mean", "540"],
        "m18": ["Cytoplasm", "Mean", "520"],
        "m19": ["Membrane", "Mean", "690"],
        "m20": ["Membrane", "Mean", "650"],
        "m21": ["Membrane", "Mean", "620"],
        "m22": ["Membrane", "Mean", "570"],
        "m23": ["Membrane", "Mean", "540"],
        "m24": ["Membrane", "Mean", "520"],
    }
    filtered_columns = []
    for k, v in patterns.items():
        for head in headers:
            if v[0] in head and v[1] in head and v[2] in head:
                filtered_columns.append(head)
    
    column_map = {}

    for k, v in patterns.items():
        for (srcname, targname) in zip(filtered_columns, target_colnames):
            if (
                v[0] in srcname
                and v[1] in srcname
                and v[2] in srcname
                and v[0] in targname
                and v[1] in targname
                and v[2] in targname
            ):
                column_map[srcname] = targname
    
    filtered_columns.append('Tissue Category')
    filtered_columns.append('Phenotype')
    target_colnames.append('Tissue Category')
    target_colnames.append('Phenotype')
    column_map['Tissue Category'] = 'Tissue Category'
    column_map['Phenotype'] = 'Phenotype'

    writer = csv.DictWriter(out, fieldnames=target_colnames)
    writer.writeheader()
    for row in reader:
        c= {}
        # print(row['Nucleus Vimentin (Opal 690) Mean (Normalized Counts, Total Weighting)'])
        for col in filtered_columns:
            c[column_map[col]] = (row[col])
        writer.writerow(c)
    

if __name__ == "__main__":
    csv_folder_unfiltered = config.args.csv_folder_unfiltered
    csv_folder = config.args.csv_folder
    
    # check if csv files which their data is unfilteres has exists
    if not csv_folder_unfiltered.exists():
        print(f"csv files with all the columns do not exists {csv_folder_unfiltered}")
        exit(-1)
    if not any(csv_folder_unfiltered.iterdir()):
        print(f"{csv_folder_unfiltered} is empty")
        exit(-1)
    
    # clean the destination folder
    if not csv_folder.exists():
        csv_folder.mkdir(exist_ok=True)
    for f in csv_folder.iterdir():
        Path.unlink(f)

    # loop over the files
    for f in csv_folder_unfiltered.iterdir():
        print(f"Processing {f}")
        convert_colnames(src=f, des=csv_folder)
    print("Finished")
