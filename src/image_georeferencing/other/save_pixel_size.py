import numpy as np
import rasterio

from rasterio.warp import calculate_default_transform
from tqdm import tqdm

def get_pixel_size(img_id, type):

    if type == "sat_est":
        type = "sat"

    tiff_path = f"/data_1/ATM/data_1/playground/georef4/tiffs/{type}/{img_id}.tif"

    try:
        with rasterio.open(tiff_path, 'r') as ds:

            # transform and get transformed width and height
            dst_crs = ds.crs
            dst_transform, dst_width, dst_height = calculate_default_transform(ds.crs, dst_crs, ds.width,
                                                                               ds.height,
                                                                               *ds.bounds)


            pix_x = round(dst_transform[0], 4)
            pix_y = round(dst_transform[4], 4)

            return np.abs(pix_x), np.abs(pix_y)

    except:
        return None, None


if __name__ == "__main__":

    import base.connect_to_db as ctd

    sql_string = "SELECT image_id, method FROM images_georef WHERE method is NOT NULL"
    data = ctd.get_data_from_db(sql_string)

    for index, row in tqdm(data.iterrows(), total=data.shape[0]):

        px, py = get_pixel_size(row['image_id'], row['method'])

        if px is None or py is None:
            continue

        sql_string = f"UPDATE images_georef SET pixel_size_x={px}, pixel_size_y={py} " \
                     f"WHERE image_id='{row['image_id']}'"
        ctd.edit_data_in_db(sql_string, add_timestamp=False, catch=False)
