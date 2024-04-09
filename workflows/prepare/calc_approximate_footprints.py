# Package imports
from tqdm import tqdm

# Custom imports
import src.base.connect_to_database as ctd
import src.prepare.calc_approximate_footprint as caf

# Display imports
import src.display.display_shapes as ds

# CONSTANTS
ADAPT_WITH_REMA = True

# Variables
overwrite = False

# Debug variables
debug_display_footprint = False


def calc_approximate_footprints():
    """
    Calculate the approximate footprints for all images in the database. If the
    overwrite flag is set to False, only images without a footprint are updated.
    Args:
    Returns:
    """

    # establish connection to psql
    conn = ctd.establish_connection()

    # get images and their parameters from the database
    sql_string = "SELECT image_id, azimuth, view_direction, " \
                 "x_coords, y_coords, altitude FROM images"
    data = ctd.execute_sql(sql_string, conn)

    # get the extracted parameters from the database
    sql_string_extracted = "SELECT image_id, focal_length, height, footprint_approx " \
                           "FROM images_extracted"
    data_extracted = ctd.execute_sql(sql_string_extracted, conn)

    # merge the two dataframes
    data = data.merge(data_extracted, on='image_id', how='left')

    # merge height and altitude
    data['height'] = data['height'].fillna(data['altitude'])
    data.drop(columns=['altitude'], inplace=True)

    # shuffle the data
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)

    # save how many entries are updated
    updated_entries = 0

    # filter out the images that already have a footprint
    if overwrite is False:
        data = data[data['footprint_approx'].isnull()]

    # loop over all images
    for idx, row in (pbar := tqdm(data.iterrows(), total=data.shape[0])):

        # get the image id
        image_id = row['image_id']

        pbar.set_postfix_str(f"Calc approx footprint for {image_id} "
                             f"({updated_entries} already updated)")

        # get the parameters
        center = (row['x_coords'], row['y_coords'])
        azimuth = row['azimuth']
        view_direction = row['view_direction']
        altitude = row['height']
        focal_length = row['focal_length']

        # skip if any of the parameters are None
        if azimuth is None or view_direction is None or altitude is None or focal_length is None:
            continue

        # calculate the approximate footprint
        try:
            footprint = caf.calc_approximate_footprint(center,
                                                       azimuth, view_direction,
                                                       altitude, focal_length,
                                                       ADAPT_WITH_REMA)
        except (Exception,):
            continue

        if debug_display_footprint:
            ds.display_shapes([footprint])

        # update the database
        sql_string = f"UPDATE images_extracted SET " \
                     f"footprint_approx=ST_GeomFromText('{footprint}')" \
                     f" WHERE image_id='{image_id}'"
        ctd.execute_sql(sql_string, conn)

        updated_entries += 1


if __name__ == "__main__":
    calc_approximate_footprints()
