"""
This algorithm does the following things:
- iterate all image files in the provided download folder
- checks if these files are already in the database and in the image folder
- if not adds them to the database and the image folder

Note that the images should be already extracted

"""
import sys #in order to import other python files
import os #needed for file iteration & copying

#add for import
sys.path.append('../misc')

from database_connections import *

verbose = True #specify if stuff should be printed
download_folder_path = "../../data/images/extracted"
image_folder_path = "../../data/images/images"

#copy files from one to the other folder
def copy_files(download_dir, image_dir):

    if verbose:
        print("copying files..", end='\r')

    # iterate folder and subfolders to get all files
    for root, dirs, files in os.walk(download_dir):
        for file in files:
            if file.endswith("tif"):

                source = download_dir + "/" + file
                destination = image_dir + "/" + file

                #check if file already exists in the image folder
                if not os.path.isfile(destination):

                    os.rename(source, destination)

    if verbose:
        print("copying files.. - finished")

#check if all files from the folder are in the db and add if necessary
def db_check(image_dir):

    #add the image to the images table if necessary
    def check_table_images(filename):

        #get id
        photo_id = filename.split(".")[0]

        # build filter
        filters=[{"photo_id": photo_id}]

        #get data
        count = conn.count_data("images", filters)

        #count is zero if it's not in yet
        if count > 0:
            return

        #get initial params
        file_path = image_dir + "/" + filename
        tma_number = photo_id[2:6]
        roll = photo_id[6:9]
        frame = photo_id[9:13]
        view_direction = roll[2]

        #check if the values are correct
        correct_params = True
        if not tma_number.isnumeric():
            correct_params = False
        if not frame.isnumeric():
            correct_params = False
        if view_direction not in ["V", "L", "R"]:
            correct_params = False

        #give error if something is wrong
        if correct_params == False:
            print("Something went wrong with", photo_id)
            return

        #create attributes dict
        attributes = {
            "photo_id": photo_id,
            "file_path": file_path,
            "tma_number": tma_number,
            "roll": roll,
            "frame": frame,
            "view_direction": view_direction
        }

        #add image
        conn.add_data("images", attributes)

    #add the image to the image_properties table if necessary
    def check_table_image_properties(filename):

        #get id
        photo_id = filename.split(".")[0]

        # build filter
        filters=[{"photo_id": photo_id}]

        #get data
        count = conn.count_data("image_properties", filters)

        #count is zero if it's not in yet
        if count > 0:
            return

        #create attributes dict
        attributes = {
            "photo_id": photo_id
        }

        #add image
        conn.add_data("image_properties", attributes)

    #create connection to db
    conn = Connector()

    if verbose:
        print("Start checking tables", end='\r')

    #iterate files in folder
    for filename in os.listdir(image_dir):
        if filename.endswith(".tif"):

            check_table_images(filename)
            check_table_image_properties(filename)

    if verbose:
        print("Start checking tables - finished")

if __name__ == "__main__":
    #1. Step: copy files
    copy_files(download_folder_path, image_folder_path)

    #2. Step: add to db
    db_check(image_folder_path)
