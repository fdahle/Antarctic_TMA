""" export pandas dataframe to PLY file """

import os

import numpy as np
import struct

from tqdm import tqdm

debug_print = True

def export_ply(df, file_path, overwrite=False):
    """
    Save a pandas DataFrame to a PLY file with binary_little_endian format, version 1.0.

    Parameters:
    - df: pandas DataFrame, columns should be named after the properties you want to save in the PLY file.
    - filename: str, the name of the output PLY file.
    """

    if os.path.exists(file_path) and not overwrite:
        raise FileExistsError(f"The file {file_path} already exists. "
                              f"Set overwrite=True to overwrite the file.")

    with open(file_path, 'wb') as f:
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
        for index, row in tqdm(df.iterrows(), total=df.shape[0]):
            for col in df.columns:
                dtype = df[col].dtype
                if np.issubdtype(dtype, np.floating):
                    f.write(struct.pack('<f', row[col]))
                elif np.issubdtype(dtype, np.integer):
                    f.write(struct.pack('<i', int(row[col])))
                else:
                    raise ValueError(f"Unsupported data type for column {col}: {dtype}")

    if debug_print:
        print(f"Exported {df.shape[0]} rows and {len(df.columns)} columns to {file_path}")