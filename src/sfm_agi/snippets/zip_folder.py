import os
import zipfile


def zip_folder(folder_path, output_zip_path, delete_files=False):
    print("Zipping folder...")
    print(folder_path)
    print(output_zip_path)

    # delete the output zip file if it already exists
    if os.path.exists(output_zip_path):
        print("Delete", output_zip_path)
        os.remove(output_zip_path)

    # create the zip file
    with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_STORED) as zipf:

        # iterate through all subfolders
        for folder_name, subfolders, filenames in os.walk(folder_path):

            # iterate through all files in the subfolder
            for filename in filenames:
                file_path = os.path.join(folder_name, filename)

                # Ensure the file being zipped is not the output zip file
                if os.path.abspath(file_path) != os.path.abspath(output_zip_path):
                    zipf.write(file_path, os.path.relpath(file_path, folder_path))

    # If delete_files is True, delete the unzipped files
    if delete_files:

        # iterate through all subfolders
        for folder_name, subfolders, filenames in os.walk(folder_path):
            # iterate through all files in the subfolder
            for filename in filenames:

                # Get the full path of the file
                file_path = os.path.join(folder_name, filename)

                # Check if the file is not the output zip file itself
                if os.path.abspath(file_path) != os.path.abspath(output_zip_path):
                    os.remove(file_path)

if __name__ == "__main__":

    project_name = "agi_agi_own"

    # Define the path to the folder to be zipped
    folder_path = f"/data/ATM/data_1/sfm/agi_projects/{project_name}/{project_name}.files/0/0/point_cloud/point_cloud"
    zip_path = f"/data/ATM/data_1/sfm/agi_projects/{project_name}/{project_name}.files/0/0/point_cloud/point_cloud.zip"
    delete_files = False

    # Zip the folder
    zip_folder(folder_path, zip_path, delete_files)