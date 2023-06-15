import os

#------------------------------------------------------------------------------
#   Run the program to create the data folders for the 8 codes
#------------------------------------------------------------------------------
data_folder = 'Data'
subfolders = ['iilm', 'qm', 'qmcool', 'qmdiag', 'qmidens', 'qmswitch', 'rilm', 'rilm_gauss']

if not os.path.exists(data_folder):
    os.makedirs(data_folder)

for subfolder in subfolders:
    folder_path = os.path.join(data_folder, subfolder)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)