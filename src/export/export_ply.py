""" export pandas dataframe to PLY file """

import numpy as np
import struct


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
