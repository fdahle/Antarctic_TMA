import os
import zipfile

def unzip_folder(path):
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith('.zip'):
                zip_path = os.path.join(root, file)
                extract_dir = os.path.join(root, file[:-4])  # Remove .zip extension for the folder
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
                print(f"Extracted: {zip_path} to {extract_dir}")


if __name__ == '__main__':

    project_name = "agi_tracks2"
    project_fld = f"/data/ATM/data_1/sfm/agi_projects/{project_name}/{project_name}.files/"

    unzip_folder(project_fld)