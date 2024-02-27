"""
Due to an error all the processed images in processed_images.csv were deleted.
This script restores (at least) the geo-referenced images in this file. Note that the execution time
of the images cannot be restored.
"""

import os
import src.base.modify_csv as mc

georef_type = "sat"

# path to the processed_images.csv
path_csv_file = f"/data_1/ATM/data_1/georef/{georef_type}_processed_images.csv"

# path to the geo-referenced images
path_fld_georef_images = f"/data_1/ATM/data_1/georef/{georef_type}"

for file in os.listdir(path_fld_georef_images):

    if file.endswith(".tif") is False:
        continue

    image_id = os.path.basename(file)[:-4]

    # create the dict for saving
    entry_dict = {
        "method": f"{georef_type}",
        "status": "georeferenced",
        "reason": "",
        "time": ""
    }

    print(entry_dict)

    mc.modify_csv(path_csv_file, image_id, "add", data=entry_dict, overwrite=False)



