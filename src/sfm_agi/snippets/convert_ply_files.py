import os
import zipfile

import src.load.load_pointcloud as lpc


def convert_ply_files(project_path):
    """
    Iterate all folders and zip folders in the project path and extract the
    zip folders and save them as folders with the same name.
    Then convert the ply files in all folders to readable text files
    Returns:

    """

    # Step 1: Iterate through all folders and extract zip files
    for root, _, files in os.walk(project_path):
        for file in files:
            if file.endswith('.zip'):
                zip_path = os.path.join(root, file)
                extract_dir = os.path.join(root, file[:-4])  # Remove .zip extension for the folder
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
                print(f"Extracted: {zip_path} to {extract_dir}")

    # Step 2: Find and convert all point cloud files (PLY or other formats)
    for root, _, files in os.walk(project_path):
        for file in files:
            if file.endswith('.ply') or file.endswith('.other_format_extension'):  # Add any other formats
                pointcloud_file = os.path.join(root, file)
                txt_file = os.path.join(root, file.replace('.ply', '.txt')) if file.endswith('.ply') else os.path.join(
                    root, file + '.txt')

                # Load the point cloud data (either PLY or other formats)
                point_cloud_data = lpc.load_point_cloud(pointcloud_file)

                # Convert the point cloud data to a readable text file
                with open(txt_file, 'w') as f:
                    for row in point_cloud_data:
                        f.write(" ".join(map(str, row)) + "\n")

                print(f"Converted {pointcloud_file} to {txt_file}")


if __name__ == "__main__":
    pth = "/data/ATM/data_1/sfm/agi_projects/tp_gcp_test_custom"
    convert_ply_files(pth)
