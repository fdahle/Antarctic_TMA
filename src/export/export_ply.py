""" export pandas dataframe to PLY file """

import numpy as np
import struct

debug_print = True

def export_ply(df, filename):
    """
    Save a pandas DataFrame to a PLY file with binary_little_endian format, version 1.0.

    Parameters:
    - df: pandas DataFrame, columns should be named after the properties you want to save in the PLY file.
    - filename: str, the name of the output PLY file.
    """
    with open(filename, 'wb') as f:
        # Write the PLY header
        f.write(b"ply\n")
        f.write(b"format binary_little_endian 1.0\n")
        f.write(f"element vertex {df.shape[0]}\n".encode('utf-8'))

        # Define each column as a property in the PLY file
        for col in df.columns:
            dtype = df[col].dtype
            if np.issubdtype(dtype, np.floating):
                f.write(f"property float {col}\n".encode('utf-8'))
            elif np.issubdtype(dtype, np.integer):
                f.write(f"property int {col}\n".encode('utf-8'))
            else:
                raise ValueError(f"Unsupported data type for column {col}: {dtype}")

        f.write(b"end_header\n")

        # Write the data in binary format
        for index, row in df.iterrows():
            for col in df.columns:
                dtype = df[col].dtype
                if np.issubdtype(dtype, np.floating):
                    f.write(struct.pack('<f', row[col]))
                elif np.issubdtype(dtype, np.integer):
                    f.write(struct.pack('<i', int(row[col])))
                else:
                    raise ValueError(f"Unsupported data type for column {col}: {dtype}")

    if debug_print:
        print(f"Exported {df.shape[0]} rows and {len(df.columns)} columns to {filename}")

if __name__ == "__main__":
    import os
    import pandas as pd

    project_name = "agi_agi_own"

    # Define the path to the folder to be zipped
    folder_path = f"/data/ATM/data_1/sfm/agi_projects/{project_name}/{project_name}.files/0/0/point_cloud/point_cloud"

    # iterate all txt files in the folder, load them and zip them
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.txt'):
                txt_path = os.path.join(root, file)
                ply_path = os.path.join(root, file.replace('.txt', '.ply'))

                if "tracks" in txt_path:
                    cols = ["color"]
                else:
                    cols = ["x", "y", "size", "id", "keypoint"]

                df = pd.read_csv(txt_path, delimiter=" ", header=None)
                df.columns = cols
                export_ply(df, ply_path)
                print(f"Exported {txt_path} to {ply_path}")