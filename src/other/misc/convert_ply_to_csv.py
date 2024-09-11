import os
import numpy as np
import pandas as pd
import src.load.load_pointcloud as lpc

def convert_ply_to_csv(fld):
    """
    Convert all PLY files in the given folder to CSV files using the load_point_cloud function.

    Parameters:
    - fld: str, the path to the folder containing the PLY files.
    """

    # Get all PLY files in the folder
    ply_files = [f for f in os.listdir(fld) if f.endswith('.ply')]

    # Iterate over all PLY files
    for ply_file in ply_files:
        # Full path to the PLY file
        full_path = os.path.join(fld, ply_file)

        # Load point cloud data using the provided function
        point_cloud_data = lpc.load_point_cloud(full_path)

        # Convert the numpy array to a pandas DataFrame
        df = pd.DataFrame(point_cloud_data)

        # Save the DataFrame to a CSV file without headers
        output_file = os.path.join(fld, ply_file.replace('.ply', '.csv'))
        df.to_csv(output_file, index=False, header=False)
        print(f"Saved CSV to {output_file}")

if __name__ == "__main__":
    # Define the path to the folder containing the PLY files
    ply_fld = "/home/fdahle/Desktop/agi_test2/point_cloud_small"

    # Convert the PLY files to CSV files
    convert_ply_to_csv(ply_fld)
