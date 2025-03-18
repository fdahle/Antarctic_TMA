import os
import Metashape
from shapely.geometry  import Polygon
from datetime import datetime, timezone

import src.base.connect_to_database as ctd

def save_sfm_to_db(project_path, images_paths, bbox,
                   status,
                   quality_dict=None, quality_dict_c=None,
                   status_message="", conn=None):

    if conn is None:
        conn = ctd.establish_connection()

    # create string of all images (based on list of images)
    image_names = [os.path.basename(image)[:-4] for image in images_paths]
    images_str = ",".join(image_names)

    # get current date
    timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')

    # Check if the variable is a Metashape BBox
    if isinstance(bbox, Metashape.BBox):
        # Get the corners of the bounding box
        min_corner = bbox.min
        max_corner = bbox.max

        # Convert corners to 2D coordinates (if you only need the X and Y axes)
        bbox_coords = [
            (min_corner.x, min_corner.y),
            (max_corner.x, min_corner.y),
            (max_corner.x, max_corner.y),
            (min_corner.x, max_corner.y),
            (min_corner.x, min_corner.y)  # Close the polygon
        ]

        # Create a Shapely polygon
        bbox = Polygon(bbox_coords)

        # get wkt string
        bbox = bbox.wkt
        bbox = "ST_GeomFromText('{}')".format(bbox)

    # create dict with all information
    project_dict = {
        "project_name": project_path.split("/")[-1],
        "status": status,
        "date_time": timestamp,
        "area": bbox,
        "num_images": len(image_names),
        "images": images_str,
        "message": status_message,
    }
    if quality_dict is not None:
        temp_dict = {
            "all_diff_mean": quality_dict["all_mean_difference"],
            "all_diff_std": quality_dict["all_std_difference"],
            "all_diff_abs_mean": quality_dict["all_mean_difference_abs"],
            "all_diff_abs_std": quality_dict["all_difference_abs_std"],
            "all_rmse": quality_dict["all_rmse"],
            "all_mae": quality_dict["all_mae"],
            "all_mad": quality_dict["all_mad"],
            "all_corr": quality_dict["all_correlation"],
            "mask_diff_mean": quality_dict["mask_mean_difference"],
            "mask_diff_std": quality_dict["mask_std_difference"],
            "mask_diff_abs_mean": quality_dict["mask_mean_difference_abs"],
            "mask_diff_abs_std": quality_dict["mask_difference_abs_std"],
            "mask_rmse": quality_dict["mask_rmse"],
            "mask_mae": quality_dict["mask_mae"],
            "mask_mad": quality_dict["mask_mad"],
            "mask_corr": quality_dict["mask_correlation"]
        }

        # append temp_dict to project_dict
        project_dict = {**project_dict, **temp_dict}

    if quality_dict_c is not None:
        temp_dict = {
            "c_all_diff_mean": quality_dict_c["all_mean_difference"],
            "c_all_diff_std": quality_dict_c["all_std_difference"],
            "c_all_diff_abs_mean": quality_dict_c["all_mean_difference_abs"],
            "c_all_diff_abs_std": quality_dict_c["all_difference_abs_std"],
            "c_all_rmse": quality_dict_c["all_rmse"],
            "c_all_mae": quality_dict_c["all_mae"],
            "c_all_mad": quality_dict_c["all_mad"],
            "c_all_corr": quality_dict_c["all_correlation"],
            "c_mask_diff_mean": quality_dict_c["mask_mean_difference"],
            "c_mask_diff_std": quality_dict_c["mask_std_difference"],
            "c_mask_diff_abs_mean": quality_dict_c["mask_mean_difference_abs"],
            "c_mask_diff_abs_std": quality_dict_c["mask_difference_abs_std"],
            "c_mask_rmse": quality_dict_c["mask_rmse"],
            "c_mask_mae": quality_dict_c["mask_mae"],
            "c_mask_mad": quality_dict_c["mask_mad"],
            "c_mask_corr": quality_dict_c["mask_correlation"]
        }

        # append temp_dict to project_dict
        project_dict = {**project_dict, **temp_dict}

    # remove None values
    project_dict = {k: v for k, v in project_dict.items() if v is not None}

    # Create SQL string with values embedded
    keys = ", ".join(project_dict.keys())
    # Separate how values are handled based on type (bbox as a function, others as strings)
    values = ", ".join(
        [f"{v}" if k == "area" else f"'{str(v).replace('\'', '\'\'')}'" if isinstance(v, str) else str(v)
         for k, v in project_dict.items()]
    )

    sql_string = f"INSERT INTO sfm_projects ({keys}) VALUES ({values})"

    ctd.execute_sql(sql_string, conn, add_timestamp=False)
