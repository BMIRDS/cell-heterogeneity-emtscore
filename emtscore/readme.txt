# =============================================================================
# Author: Behnaz Abdollahi
# Created by: Hassanpour's Lab
# Final modification date: July 2021
# =============================================================================
# Description:
# This module finds emtscore using predefined coefficients.
# =============================================================================
# Libraries:
# The code is written in python 3.7
# numpy, pandas, scipy 
# =============================================================================

--------------------------------------------------------------------------------------------------
Installation and setup
- You need Miniconda to be installed on your computer.
--Installing on macOS
---Step1: Copy and paste the "Miniconda3-latest-MacOSX-x86_64.sh" in a folder on your computer.
          Please find the updated version in this link.
          https://docs.conda.io/en/latest/miniconda.html
          Use Python 3.8 or above "3".
          Click on  "Miniconda3 MacOSX 64-bit bash" to download bash file and follow Step2 instruction 
          for installation.
          or 
          Click on "Miniconda3 MacOSX 64-bit pkg" to install the ios package and install it like 
          other packages in mac.
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
Folder structure of emtscore project.
-data
    
    --testdata_csv
        --should includes .csv files for example
        'TSA3 159P_[10266,56380]_cell_seg_data.csv'
        'TSA3 159P_[10420,50489]_cell_seg_data.csv'
    --testdata_csv_unfiltered
        --should include .csv files and have all the columns names that are in txt file
        'TSA3 159P_[10266,56380]_cell_seg_data.csv' 
        'TSA3 159P_[10420,50489]_cell_seg_data.csv'
    --testdata_txt
        --should includes .txt files example
        'TSA3 159P_[10266,56380]_cell_seg_data.txt'
        'TSA3 159P_[10420,50489]_cell_seg_data.txt'
-code
    -config.py
    -emtscore.py

-result
    -scores.csv 
    for example it should be like as below

    name,score
    "TSA3 sc4_[13723,47608]_cell_seg_data.csv",0.3659
    "TSA3 sc4_[11808,45003]_cell_seg_data.csv",0.2159
    
--------------------------------------------------------------------------------------------------
Finding emtscore of new images.

Step0: copy text files in folder ../data/testdata_txt/
Step1: open terminal window and type
       -activate conda myenv 
       then hit enter and type
       -cd emtscore/code/

Step2: convert txt files into csv files
    -python convert_txt2csv.py
    or
    -python convert_txt2csv.py --txt_folder=../data/testdata_txt --csv_folder=../data/testdata_csv_unfiltered

Step3: Filtering columns used in the analysis and updating their names.
        -python convert_cols.py
        or
        -python convert_cols.py --src=../data/testdata_csv_unfiltered --des=../data/testdata_csv


Note:Reassure that folder data/testdata_csv is created and .csv files are created.
Step4:
    python emtscore.py 
    or
    python emtscore.py --testdata_folder data/testdata_csv 

Step5: Inside folder "result", you should have the file "scores.csv"
--------------------------------------------------------------------------------------------------
Notes regarding the column values.
Columns required with the same name in the input data:
--Tissue Category
--Phenotype
    --Phenotypes name should be as follow:
    ----"vim only"
    ----"Ecad only"
    ----"K8ecad"
    ----"K8"
    ----"K14"
    ----"K8vim"
    ----"K14vim"
    ----"Trip+"
    ----"Snail"
    ----"vimzeb"
--------------------------------------------------------------------------------------------------
If you need to change markers coefficients you can update them in config.py.
--parser.add_argument("--Ecadonly", type=int, default=-3) 
--for example in the above line you can update the default value from -3 to any positive or 
negative integer number
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
save config.py file and Done!
--------------------------------------------------------------------------------------------------
