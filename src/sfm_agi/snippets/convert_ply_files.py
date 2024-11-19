import os

import src.load.load_ply as lp
import src.sfm_agi.snippets.unzip_folder as uzf

def convert_ply_files(project_path, unzip=True):
    """
    Iterate all folders and zip folders in the project path and extract the
    zip folders and save them as folders with the same name.
    Then convert the ply files in all folders to readable text files
    Returns:

    """

    # Step 1: Iterate through all folders and extract zip files
    if unzip:
        uzf.unzip_folder(project_path)

    # Step 2: Find and convert all point cloud files (PLY or other formats)
    for root, _, files in os.walk(project_path):
        for file in files:
            if file.endswith('.ply') or file.endswith('.other_format_extension'):  # Add any other formats
                ply_file = os.path.join(root, file)
                txt_file = os.path.join(root, file.replace('.ply', '.txt')) if file.endswith('.ply') else os.path.join(
                    root, file + '.txt')

                # Load the ply data
                ply_data = lp.load_ply(ply_file)

                # Convert the point cloud data to a readable text file
                with open(txt_file, 'w') as f:
                    for row in ply_data:
                        f.write(" ".join(map(str, row)) + "\n")

                print(f"Converted {ply_file} to {txt_file}")


if __name__ == "__main__":
    project_name = "agi_tracks2"
    pth = f"/data/ATM/data_1/sfm/agi_projects/{project_name}"
    convert_ply_files(pth, False)
