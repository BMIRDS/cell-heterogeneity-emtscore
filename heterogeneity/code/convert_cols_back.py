import collections
import config
import csv
import pandas as pd
import re
from pathlib import Path


def convert_colnames(src, des):
    standard_cols = collections.defaultdict(list)
    target_colnames = [
        "Nucleus Vim (Opal 690) Mean (Normalized Counts, Total Weighting)",
        "Nucleus Ecad (Opal 650) Mean (Normalized Counts, Total Weighting)",
        "Nucleus Snail (Opal 620) Mean (Normalized Counts, Total Weighting)",
        "Nucleus ZEB1 (Opal 570) Mean (Normalized Counts, Total Weighting)",
        "Nucleus K8 (Opal 540) Mean (Normalized Counts, Total Weighting)",
        "Nucleus K14 (Opal 520) Mean (Normalized Counts, Total Weighting)",
        "Entire Cell Vim (Opal 690) Mean (Normalized Counts, Total Weighting)",
        "Entire Cell Ecad (Opal 650) Mean (Normalized Counts, Total Weighting)",
        "Entire Cell Snail (Opal 620) Mean (Normalized Counts, Total Weighting)",
        "Entire Cell ZEB1 (Opal 570) Mean (Normalized Counts, Total Weighting)",
        "Entire Cell K8 (Opal 540) Mean (Normalized Counts, Total Weighting)",
        "Entire Cell K14 (Opal 520) Mean (Normalized Counts, Total Weighting)",
        "Cytoplasm Vim (Opal 690) Mean (Normalized Counts, Total Weighting)",
        "Cytoplasm Ecad (Opal 650) Mean (Normalized Counts, Total Weighting)",
        "Cytoplasm Snail (Opal 620) Mean (Normalized Counts, Total Weighting)",
        "Cytoplasm ZEB1 (Opal 570) Mean (Normalized Counts, Total Weighting)",
        "Cytoplasm K8 (Opal 540) Mean (Normalized Counts, Total Weighting)",
        "Cytoplasm K14 (Opal 520) Mean (Normalized Counts, Total Weighting)",
        "Membrane Vim (Opal 690) Mean (Normalized Counts, Total Weighting)",
        "Membrane Ecad (Opal 650) Mean (Normalized Counts, Total Weighting)",
        "Membrane Snail (Opal 620) Mean (Normalized Counts, Total Weighting)",
        "Membrane ZEB1 (Opal 570) Mean (Normalized Counts, Total Weighting)",
        "Membrane K8 (Opal 540) Mean (Normalized Counts, Total Weighting)",
        "Membrane K14 (Opal 520) Mean (Normalized Counts, Total Weighting)",
    ]
    
    # f = open("1050A TSA_Core[1,1,B]_[7991,39736]_cell_seg_data.csv", "r")
    # f = open('TSA3 sc8_[17137,50417]_cell_seg_data.csv', 'r')
    f = open(src, "r")
    out = open(Path(des, file_path.name), "w", newline="")

    reader = csv.DictReader(f)
    headers = reader.fieldnames

    keywords = ["Nucleus", "Mean", "Vim"]
    patterns = {
        "m1": ["Nucleus", "Mean", "Vim"],
        "m2": ["Nucleus", "Mean", "Ecad"],
        "m3": ["Nucleus", "Mean", "Snail"],
        "m4": ["Nucleus", "Mean", "ZEB1"],
        "m5": ["Nucleus", "Mean", "K8"],
        "m6": ["Nucleus", "Mean", "K14"],
        "m7": ["Entire", "Mean", "Vim"],
        "m8": ["Entire", "Mean", "Ecad"],
        "m9": ["Entire", "Mean", "Snail"],
        "m10": ["Entire", "Mean", "ZEB1"],
        "m11": ["Entire", "Mean", "K8"],
        "m12": ["Entire", "Mean", "K14"],
        "m13": ["Cytoplasm", "Mean", "Vim"],
        "m14": ["Cytoplasm", "Mean", "Ecad"],
        "m15": ["Cytoplasm", "Mean", "Snail"],
        "m16": ["Cytoplasm", "Mean", "ZEB1"],
        "m17": ["Cytoplasm", "Mean", "K8"],
        "m18": ["Cytoplasm", "Mean", "K14"],
        "m19": ["Membrane", "Mean", "Vim"],
        "m20": ["Membrane", "Mean", "Ecad"],
        "m21": ["Membrane", "Mean", "Snail"],
        "m22": ["Membrane", "Mean", "ZEB1"],
        "m23": ["Membrane", "Mean", "K8"],
        "m24": ["Membrane", "Mean", "K14"],
    }
    filtered_columns = []
    for k, v in patterns.items():
        for head in headers:
            if v[0] in head and v[1] in head and v[2] in head:
                filtered_columns.append(head)
    # print(len(filtered_columns))
    column_map = {}

    writer = csv.DictWriter(out, fieldnames=target_colnames)
    writer.writeheader()

    for k, v in patterns.items():
        for (src, targ) in zip(filtered_columns, target_colnames):
            if (
                v[0] in src
                and v[1] in src
                and v[2] in src
                and v[0] in targ
                and v[1] in targ
                and v[2] in targ
            ):
                column_map[src] = targ
    
    for row in reader:
        c= {}
        # print(row['Nucleus Vimentin (Opal 690) Mean (Normalized Counts, Total Weighting)'])
        for col in filtered_columns:
            c[column_map[col]] = (row[col])
        writer.writerow(c)


if __name__ == "__main__":
    csv_folder_unfiltered = config.args.csv_folder_unfiltered
    csv_folder = config.args.csv_folder
    for f in csv_folder_unfiltered.iterdir():
        convert_colnames(src=csv_folder_unfiltered, des=csv_folder)
