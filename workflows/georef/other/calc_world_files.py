# Library imports
import numpy as np
import os
from tqdm import tqdm

# Local imports
import src.export.export_world_file as ewf

# Variables
overwrite = False
path_input_fld = "/data/ATM/data_1/georef/sat"


def calc_world_files():

    # get all tiff files in the folder
    all_files = os.listdir(path_input_fld)
    all_files = [file for file in all_files if file.endswith(".tif")]

    # remove ending
    all_files = [file[:-4] for file in all_files]

    # iterate over all files
    for file in (pbar := tqdm(all_files)):

        if os.path.exists(f"{path_input_fld}/{file}.tfw") and overwrite is False:
            continue

        # update the progress bar
        pbar.set_postfix_str(f"Calculate world file for {file}")

        try:
            # get the transformation matrix as a numpy array
            trans_matrix = np.loadtxt(f"{path_input_fld}/{file}_transform.txt")
        except (Exception,):
            continue

        # convert to a 3x3 matrix
        trans_matrix = trans_matrix.reshape(3, 3)

        # export the world file
        ewf.export_world_file(trans_matrix, f"{path_input_fld}/{file}.tfw")


if __name__ == "__main__":
    calc_world_files()
