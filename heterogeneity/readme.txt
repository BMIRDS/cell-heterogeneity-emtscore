# =============================================================================
# Author: Behnaz Abdollahi
# Created by: Hassanpour's Lab
# Final modification date: July 2021
# =============================================================================
# Description:
# This module finds heterogeneity label of images.
# Train and test includes in this module.
# =============================================================================
# Libraries:
# The code is written in python 3.7
# numpy, pandas, scipy 
# =============================================================================

--------------------------------------------------------------------------------------------------
Installation and setup
- You need Miniconda to be installed on your computer.
--Installing on macOS
---Step1: Copy and paste the "Miniconda3-latest-MacOSX-x86_64.sh" in a folder on your laptop.
          Please find the updated version in this link.
          https://docs.conda.io/en/latest/miniconda.html
          Use Python 3.8 or above "3".
          Click on  "Miniconda3 MacOSX 64-bit bash" to download bash file and see Step2 instruction 
          for installation.
          or 
          Click on "Miniconda3 MacOSX 64-bit pkg" to install the ios package and install it 
          the same as other packages in mac.
--Step2: Open terminal window, go to the folder that the .bash file is copied and type:
    -bash Miniconda3-latest-MacOSX-x86_64.sh
    -Accept the defaults and press enter or y for all the questions.
--Step3: You need to create a virtual environment.
         Step3-1:
            Open terminal window and type 
            -conda create --name myenv 
            (choose any name instead of myenv, the rest of the command is typed as it is)
            proceed ([y]/n)? (type y)
        Step3-2: 
            Close terminal window and open a new one, then type.
            -conda activate myenv 
            (if you have chosen another name then write it instead of myenv)
--Step4: If conda virtual environment created successfully, you should see the name of it next
to your cursor on terminal. For example:
            (myenv) $

(For any other information in terms of conda installation please see the link below.
https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands)

-Once you have installed and activated conda, you need to install several other libraries. To install any 
python library you need to open terminal window and activate the conda virtual environment.
(Please see Step3 and Step 4)

-conda install -c anaconda scipy
-conda install numpy
-conda install pandas

    -Note: If you get an error when running a code in terminal and the error is like for example:
    "Import Error: No module named numpy"
    This error indicates that numpy is not installed in your computer. So you can type in terminal window.
    -conda install numpy 
    -If asked proceed ([y]/n)? (type y)
    If the error is an import error for any other libary other than numpy then google it with
    keyword "library name "conda" "install" then copy and paste the installation command in terminal.
    _If the import error is still raised after successfully installed the module, then Close the 
    terminal window and open it again and activate the virtual environment and run the code.

--------------------------------------------------------------------------------------------------
Running python codes in terminal window.
-To run any python code(files with .py extension), please do these steps:
-Step1: Open terminal window
-Step2: activate conda myenv 
    Note:(if virtual environment is not created please read installation and set up.
Step3: Go to the folder that python file is copied. For example in this project type:
    -cd heterogeneity
    -cd code
    -python test.py 
    or 
    please see "Finding heterogeneity label of new images" section in this document.
--------------------------------------------------------------------------------------------------
Folder structure of heterogeneity project.
-data
    
    --testdata_txt
    --testdata_csv_unfiltered
    --testdata_csv
    --traindata_txt
    --traindata_csv_unfiltered
    --traindata_csv
    --label
-code
    -config.py
    -data.py
    -test.py
    -train.py
-model
    -modelname.pkl
    -transformname.pkl
-result
    -test_res.csv
--------------------------------------------------------------------------------------------------
Finding heterogeneity label of new images using test.

Step0: copy text files into the folder ../data/testdata_txt/
Step1: open terminal window and type
       -activate conda myenv 
       then hit enter and type
       -cd heterogeneity/code/
Step2: convert txt files into csv files
    python convert_txt2csv.py
    or
    python convert_txt2csv.py --txt_folder=../data/testdata_txt --csv_folder=../data/testdata_csv_unfiltered

Step3: Filtering columns used in the analysis and updating their names.
        -python convert_cols.py
        or
        -python convert_cols.py --src=../data/testdata_csv_unfiltered --des=../data/testdata_csv

Note:Reassure that folder data/testdata_csv is created and .csv files are created.

Step4: 
    python test.py 
    or
    python test.py --testdata_folder data/testdata_csv --model_name model_...pkl
    --model_path model --result_folder result


Step5: Inside folder "result", you should have test_res.csv.

--------------------------------------------------------------------------------------------------
Training new model with logistic regression.

Step0: copy text files into the folder ../data/traindata_txt/

Step1: open terminal window and type
       -activate conda myenv 
       then hit enter and type
       -cd heterogeneity/code/
Step2: convert txt files into csv files
    python convert_txt2csv.py --txt_folder=../data/traindata_txt --csv_folder=../data/trindata_csv_unfiltered

Step3: Filtering columns used in the analysis and updating their names.
        -python convert_cols.py --src=../data/traindata_csv_unfiltered --des=../data/traindata_csv

Step2: create a .csv file of labels with two columns. ("name", "label")
       column "name" is the image name without ".csv" so for example rows should be saved like:
        name, label
        "TSA3 149P_[10214,53328]",mid
        "TSA3 149P_[13869,48115]",high
        "TSA3 159P_[13840,56322]",low
Step3: 
        python train.py
        or
        python code/train.py --traindata_folder data/traindata_csv --label_file data/label/label.csv --split_ratio 0.3 --model_path model
Note: Trained model is saved under folder model/model_....pkl and model/transform_....pkl
       for example the model folder includes:
       model_07_13_2021.pkl and transform_07_13_2021.pkl
       Transform data is saved under mode/transform_....pkl
       postfix (_07_13_2021) are the same only prefix of one of the pickle files are "model" and "transform"

--------------------------------------------------------------------------------------------------
We did several experimenal analysis on several feature vectors and finally only those columns
related to the mean information of cell markers are included in the calculations.
Columns required to be included with the same name:
- Columns required to run the code:

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

--Tissue Category
--Sample Name(it should have overlapping with the file name)

--------------------------------------------------------------------------------------------------previous name columns not used in new code
#--"Nucleus Opal 690 Mean (Normalized Counts, Total Weighting)"
#--"Nucleus Ecad (Opal 650) Mean (Normalized Counts, Total Weighting)"
#--"Nucleus Snail (Opal 620) Mean (Normalized Counts, Total Weighting)"
#--"Nucleus ZEB1 (Opal 570) Mean (Normalized Counts, Total Weighting)"
#--"Nucleus K8 (Opal 540) Mean (Normalized Counts, Total Weighting)"
#--"Nucleus K14 (Opal 520) Mean (Normalized Counts, Total Weighting)"
#
#--"Entire Cell Vim (Opal 690) Mean (Normalized Counts, Total Weighting)"
#--"Entire Cell Ecad (Opal 650) Mean (Normalized Counts, Total Weighting)"
#--"Entire Cell Snail (Opal 620) Mean (Normalized Counts, Total Weighting)"
#--"Entire Cell ZEB1 (Opal 570) Mean (Normalized Counts, Total Weighting)"
#--"Entire Cell K8 (Opal 540) Mean (Normalized Counts, Total Weighting)"
#--"Entire Cell K14 (Opal 520) Mean (Normalized Counts, Total Weighting)"
#
#--"Cytoplasm Vim (Opal 690) Mean (Normalized Counts, Total Weighting)"
#--"Cytoplasm Ecad (Opal 650) Mean (Normalized Counts, Total Weighting)"
#--"Cytoplasm Snail (Opal 620) Mean (Normalized Counts, Total Weighting)"
#--"Cytoplasm ZEB1 (Opal 570) Mean (Normalized Counts, Total Weighting)"
#--"Cytoplasm K8 (Opal 540) Mean (Normalized Counts, Total Weighting)"
#--"Cytoplasm K14 (Opal 520) Mean (Normalized Counts, Total Weighting)"
#
#--"Membrane Vim (Opal 690) Mean (Normalized Counts, Total Weighting)"
#--"Membrane Ecad (Opal 650) Mean (Normalized Counts, Total Weighting)"
#--"Membrane Snail (Opal 620) Mean (Normalized Counts, Total Weighting)"
#--"Membrane ZEB1 (Opal 570) Mean (Normalized Counts, Total Weighting)"
#--"Membrane K8 (Opal 540) Mean (Normalized Counts, Total Weighting)"
#--"Membrane K14 (Opal 520) Mean (Normalized Counts, Total Weighting)"
#
