from tqdm import tqdm

import src.base.connect_to_database as ctd
import src.load.load_shape_data as lsd


# Constants
PATH_SHP_FILE = "/data_1/ATM/data_1/shapefiles/TMA_Photocenters/TMA_pts_20100927.shp"

# Variables
path_images = "/data_1/ATM/data_1/aerial/TMA/downloaded"  # path to the images
_overwrite = False  # overwrite existing entries
_catch = False  # catch exceptions


def add_tma_shp_data(path_shp_file, overwrite=False, catch=False):
    """
    Updates database entries with data loaded from a TMA shapefile.

    This function iterates through each image ID in the database, retrieves corresponding
    data from a shapefile, and updates the database entries with this data. It supports
    overwriting existing entries and catching exceptions during the update process.

    Args:
        path_shp_file: The file path to the TMA shapefile containing the data to be loaded.
        overwrite: A flag indicating whether existing database entries should be overwritten.
            Defaults to False.
        catch: A flag indicating whether exceptions during the update process should be caught.
            Defaults to False.

    Raises:
        Exception: Propagates any exceptions encountered during database update operations if
            `catch` is False.
    """

    # load the content of the tma shape file
    shape_data = lsd.load_shape_data(path_shp_file)

    # connect to the database
    conn = ctd.establish_connection()

    # get all image_ids from the images table
    sql_string = "SELECT image_id FROM images"
    image_data = ctd.execute_sql(sql_string, conn)
    image_ids = image_data['image_id'].values.tolist()

    for image_id in (pbar := tqdm(image_ids)):

        pbar.set_postfix_str(f"update {image_id}")

        # get the entity image_id from the image image_id (it's almost the same,
        # just without the view_direction)
        entity_id = image_id[:6] + image_id[-4:]

        # get the data for this entity_id
        shape_row = shape_data[shape_data["ENTITY_ID"] == entity_id].to_dict(orient='records')[0]

        # get the image params for this image_id
        image_path = "/data_1/ATM/data_1/aerial/TMA/downloaded/" + image_id + ".tif"
        tma = image_id[2:6]
        roll = image_id[6:9]
        frame = int(image_id[9:13])
        view_direction = image_id[8]
        altitude = float(shape_row["Altitude"])
        azimuth = float(shape_row["Azimuth_dd"])
        date_of_recording = str(shape_row["Date_"]).rstrip('\n')
        cam_id = int(shape_row["v_cam"])
        location = str(shape_row["Location"])
        point_x = float(shape_row["POINT_X"])
        point_y = float(shape_row["POINT_Y"])
        point = f"ST_GeomFromText('POINT({point_x} {point_y})',3031)"

        # get the geometry
        geom = str(shape_row["geometry"])
        x_coords = geom[7:].split(" ")[0]
        y_coords = geom[:-1].split(" ")[2]

        # fallback if the date cannot be recognized
        date_year = 0
        date_month = 0
        date_day = 0

        # the date_ field is something messy, so we need different methods to handle it
        if len(date_of_recording) == 4:  # only year available
            date_year = int(date_of_recording)
            date_month = 0
            date_day = 0
        elif date_of_recording.count('/') == 2:  # we have all things
            date_year = int(date_of_recording.split("/")[2])
            date_month = int(date_of_recording.split("/")[0])
            date_day = int(date_of_recording.split("/")[1])
        elif date_of_recording.count('/') == 1:
            # super strange combi with two years shown in date
            if date_of_recording[4] == "/" and \
                    len(date_of_recording.split("/")[0]) == 4 and \
                    len(date_of_recording.split("/")[1]) == 2:
                date_year = int(date_of_recording.split("/")[0])
                date_month = 0
                date_day = 0
            # only month and year
            if len(date_of_recording.split("/")[1]) == 4:
                date_year = int(date_of_recording.split("/")[1])
                date_month = int(date_of_recording.split("/")[0])
                date_day = 0

        # save all params for updating
        update_dict = {
            "path_file": [image_path, "string"],
            "tma_number": [tma, "integer"],
            "roll": [roll, "string"],
            "frame": [frame, "integer"],
            "view_direction": [view_direction, "string"],
            "altitude": [altitude, "float"],
            "azimuth": [azimuth, "float"],
            "date_of_recording": [date_of_recording, "string"],
            "date_year": [date_year, "integer"],
            "date_month": [date_month, "integer"],
            "date_day": [date_day, "integer"],
            "cam_id": [cam_id, "integer"],
            "location": [location, "string"],
            "x_coords": [x_coords, "float"],
            "y_coords": [y_coords, "float"],
            "point_x": [point_x, "float"],
            "point_y": [point_y, "float"],
            "point": [point, "geom"]
        }

        set_clauses = []

        for key, (value, value_type) in update_dict.items():

            if value is None:
                continue  # Skip null updates
            if value_type == "string":
                value = f"'{value}'"
            else:
                value = str(value)

            # Construct SET clause with COALESCE
            if overwrite:
                set_clause = f"{key} = {value}"
            else:
                set_clause = f"{key} = COALESCE({key}, {value})"
            set_clauses.append(set_clause)

        set_clause_str = ", ".join(set_clauses)
        sql_string = f"UPDATE images SET {set_clause_str} WHERE image_id = '{image_id}'"

        try:
            ctd.execute_sql(sql_string, conn)
        except (Exception,) as e:
            if catch:
                pbar.set_postfix_str(f"{image_id} failed")
            else:
                raise e
        pbar.set_postfix_str(f"{image_id} updated")


if __name__ == "__main__":

    add_tma_shp_data(PATH_SHP_FILE, _overwrite, _catch)
