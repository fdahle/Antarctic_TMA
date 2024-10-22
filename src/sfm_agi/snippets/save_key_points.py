import os
import numpy as np
import zipfile
from plyfile import PlyData

import src.load.load_image as li
import src.display.display_images as di


def save_key_points(image_ids, point_cloud_fld, kp_folder):

    # define path to the zip file
    path_zip = os.path.join(point_cloud_fld, "point_cloud.zip")

    ply_dict = {}

    with zipfile.ZipFile(path_zip, 'r') as zip_ref:
        # List all files in the zip
        all_files = zip_ref.namelist()

        # Filter and process only PLY files
        for file in all_files:
            if file.endswith(".ply"):
                # Open the PLY file within the zip
                with zip_ref.open(file, 'r') as ply_file:

                    # Extract the filename without path to use as key
                    filename = os.path.basename(file)

                    # ignore tracks ply
                    if filename == "tracks.ply":
                        continue

                    # ignore points ply
                    if filename.startswith("points"):
                        continue

                    print("Extract coords for", filename)

                    # Load the PLY file data using plyfile
                    ply_data = PlyData.read(ply_file)

                    # Assuming 'vertex' element is present and has 'x' and 'y' properties
                    vertex = ply_data['vertex']
                    x = vertex['x']
                    y = vertex['y']

                    # Create numpy array from the x, y data
                    coords = np.column_stack((x, y))

                    # convert to int
                    coords = coords.astype(int)

                    # Store the content in the dictionary
                    ply_dict[filename] = coords

    # Load the images
    for i, key in enumerate(ply_dict.keys()):

        image_id = image_ids[i]

        # Load the image
        img = li.load_image(image_id)

        # define save path
        save_path = f"{kp_folder}/{image_id}"

        # Display the image (and save it)
        di.display_images(img, points=[ply_dict[key]], save_path=save_path)
