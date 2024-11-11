
project_name = "tst_gc"

# get path to project dem
hist_dem_path = f"/data/ATM/data_1/sfm/agi_projects/{project_name}/output/{project_name}_dem_relative.tif"
transform_path = f"/data/ATM/data_1/sfm/agi_projects/{project_name}/data/georef/transformation.txt"

# load the dem
import src.load.load_image as li
dem = li.load_image(hist_dem_path, image_type="dem")

print(dem.shape)