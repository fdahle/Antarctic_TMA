""" create a new AgiProject """
base_fld = "/data/ATM/data_1/sfm/agi_projects2"

# define the input data
mode = "glaciers"   # "glaciers" or "images"

input_glaciers = ["Crane Glacier"]

#project_name = "test"
#input_images = ["CA215332V0419", "CA215332V0420", "CA215332V0421", "CA215332V0422"]

from shapely import wkt
import src.base.connect_to_database as ctd
import src.load.load_image as li
import src.load.load_military_focal_length as lmfl
import src.sfm_agi2.other.get_images_for_glacier as gfi

# load the Agi class
from src.sfm_agi2 import AgiProject as AgP

# connect to database
conn = ctd.establish_connection()

# default values for substitution
default_focal_lengths = {'V': 152.07, 'L': 152.78, 'R': 152.29}

# create fake glacier project for input images
if mode == "images":
    input_glaciers = [project_name]  # noqa

for glacier in input_glaciers:  # noqa

    # for glaciers, we need to get the images and adapt the glacier name
    if mode == "glaciers":
        input_images, glac_bounds = gfi.get_images_for_glacier(glacier,
                                                               return_bounds=True,
                                                               conn=conn)
        project_name = glacier.replace(" ", "_").lower()
    else:
        glac_bounds = None

    # create optional dicts
    focal_length_dict = {}
    rotation_dict = {}
    footprint_dict = {}

    # get image attributes
    for img_id in input_images:  # noqa

        # get information for that image
        sql_string = (f"SELECT focal_length, azimuth_exact, "
                      f" ST_AsText(footprint_exact) as footprint"
                      f" FROM images_extracted WHERE image_id='{img_id}'")
        data = ctd.execute_sql(sql_string, conn)

        # get focal length and
        focal = data['focal_length'][0]

        # try to get the focal length from the military focal length table
        if focal is None:
            focal = lmfl.load_military_focal_length(img_id, conn=conn)

        # add default value to focal length
        if focal is None:
            for key in default_focal_lengths:
                if key in img_id:
                    focal = default_focal_lengths[key]
                    break
            else:
                raise ValueError(f"No focal length found and no match in defaults for image_id {img_id}")
        focal_length_dict[img_id] = focal

        # get rotation (yaw, pitch roll)
        yaw = data['azimuth_exact'][0]
        yaw = 360 - yaw + 90  # account for the different coordinate system
        yaw = round(yaw, 2)
        yaw = yaw % 360
        pitch = 0
        if "V" in img_id:
            roll = 0
        elif "L" in img_id:
            roll = 30
        elif "R" in img_id:
            roll = 360 - 30
        else:
            raise ValueError(f"Unknown image direction {img_id}")
        rotation_dict[img_id] = (yaw, pitch, roll)

        # load the footprint
        footprint_dict[img_id] = wkt.loads(data['footprint'][0])

        # just to make sure that the image is copied from backup folder to download folder
        li.load_image(img_id)

    # create the project
    ap = AgP.AgiProject(project_name, base_fld,
                        overwrite=True, resume=False,
                        debug=True)
    ap.set_input_images(input_images)

    # set optional data
    ap.set_optional_data(
        absolute_bounds=glac_bounds,
        mask_src_folder="/data/ATM/data_1/aerial/TMA/masked",
        camera_rotations=rotation_dict,
        camera_footprints=footprint_dict,
        focal_lengths=focal_length_dict,
    )

    # run the project
    ap.run_project()