"""convert ply files in a folder to text files"""

# Library imports
import os

# Local imports
import src.load.load_ply as lp
import src.sfm_agi.snippets.unzip_folder as uzf

def convert_ply_files(project_path: str, unzip: bool = True) -> None:
    """
     Extracts all zip files within a project directory and converts point cloud  (ply) files
     into readable text files for easier processing.
    Args:
        project_path (str): Path to the project directory containing files and folders.
        unzip (bool, optional): Whether to unzip compressed folders before processing. Defaults to True.
    Returns:
        None
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
                try:
                    ply_data = lp.load_ply(ply_file)
                except Exception as e:
                    print(f"Error loading {ply_file}: {e}")
                    continue

                # Convert the point cloud data to a readable text file
                with open(txt_file, 'w') as f:
                    for row in ply_data:
                        f.write(" ".join(map(str, row)) + "\n")

                print(f"Converted {ply_file} to {txt_file}")