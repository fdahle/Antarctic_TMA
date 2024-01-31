import geopandas as gpd
import os
import pandas as pd

from shapely.geometry import LineString

import image_georeferencing.sub.calc_camera_position as ccp
import image_georeferencing.sub.convert_image_to_footprint as citf

import base.connect_to_db as ctd
import base.load_image_from_file as liff
import base.print_v as p

debug_save_exact_footprint_db = True
debug_save_exact_footprints_shape = True
debug_save_exact_photocenter_db = True
debug_save_exact_photocenters_shape = True
debug_save_error_vector_db = True
debug_save_error_vector_shape = True


def save_footprint(image_id, footprint_type, shp_data=None,
                   overwrite=False, catch=True, verbose=False, pbar=None):

    p.print_v(f"Save {image_id} in db", verbose=verbose, pbar=pbar)

    if footprint_type == "satellite":
        fld_georeferenced_images = "/data_1/ATM/data_1/playground/georef2/tiffs/sat"
        fld_shapefiles = "/data_1/ATM/data_1/playground/georef2/overview/sat/sat_"
    elif footprint_type == "image":
        fld_georeferenced_images = "/data_1/ATM/data_1/playground/georef2/tiffs/img"
        fld_shapefiles = "/data_1/ATM/data_1/playground/georef2/overview/img/img_"

    img_path = fld_georeferenced_images + "/" + image_id + ".tif"

    # first check if we have a georeferenced image
    if os.path.exists(img_path) is False:
        return

    # already get the db data for this image
    sql_string = "SELECT image_id, footprint, position_exact, position_error_vector " \
                 f"From images_extracted WHERE image_id='{image_id}'"
    data = ctd.get_data_from_db(sql_string, catch=catch, verbose=verbose, pbar=pbar)
    data = data.iloc[0]

    # load the georeferenced image
    img, transform = liff.load_image_from_file(img_path, return_transform=True,
                                               catch=catch, verbose=verbose, pbar=pbar)

    # convert image to approx_footprint
    footprint_exact = citf.convert_image_to_footprint(img, image_id, transform,
                                                     catch=catch, verbose=verbose, pbar=pbar)

    if debug_save_exact_footprint_db:

        # don't save if we don't want to overwrite and we already have an entry
        if overwrite is True or data['footprint'] is None:

            sql_string = f"UPDATE images_extracted SET " \
                         f"footprint=ST_GeomFromText('{footprint_exact}'), " \
                         f"footprint_exact_type='{footprint_type}' " \
                         f"WHERE image_id='{image_id}'"
            ctd.edit_data_in_db(sql_string, catch=catch, verbose=verbose, pbar=pbar)

    # save the exact approx_footprint in a shapefile if we want
    if debug_save_exact_footprints_shape:
        shape_path = fld_shapefiles + "footprints.shp"

        # Create a GeoDataFrame with the approx_footprint
        footprint_gdf = gpd.GeoDataFrame(geometry=[footprint_exact])
        footprint_gdf.crs = "EPSG:3031"
        footprint_gdf['image_id'] = image_id

        if shp_data is not None:
            footprint_gdf['nr_points'] = shp_data["nr_points"]
            footprint_gdf['conf'] = shp_data["conf"]
            footprint_gdf['sat_min_x'] = shp_data['sat_min_x']
            footprint_gdf['sat_max_x'] = shp_data['sat_max_x']
            footprint_gdf['sat_min_y'] = shp_data['sat_min_y']
            footprint_gdf['sat_max_y'] = shp_data['sat_max_y']

        # Create the shapefile if it doesn't exist
        if not os.path.exists(shape_path):

            footprint_gdf.to_file(shape_path)

        else:
            # Read the existing shapefile
            existing_gdf = gpd.read_file(shape_path)

            # check if the image_id is already in our existing footprints
            image_id = footprint_gdf.iloc[0]['image_id']

            # Update the entries in existing_gdf
            if image_id in existing_gdf['image_id'].values:

                # which columns should be updated
                columns_to_update = ["geometry"]
                if shp_data is not None:
                    columns_to_update = columns_to_update + ["nr_points", "avg_conf",
                                                             "sat_min_x", "sat_max_x",
                                                             "sat_min_y", "sat_max_y"]

                # filter for this exact image_id
                filter_condition = existing_gdf['image_id'] == image_id
                existing_gdf.loc[filter_condition, columns_to_update] = footprint_gdf.iloc[0][columns_to_update]

            # Append the new approx_footprint to the existing GeoDataFrame
            else:
                existing_gdf = pd.concat([existing_gdf, footprint_gdf], ignore_index=True)

            # Save the updated GeoDataFrame to the shapefile
            existing_gdf.to_file(shape_path)

    # calculate the photocenter
    x_exact, y_exact = ccp.calc_camera_position(footprint_exact)

    if x_exact is None or y_exact is None:
        p.print_v(f"Position for {image_id} is invalid", verbose=verbose, color="red", pbar=pbar)
        return

    if debug_save_exact_photocenter_db:

        # don't save if we don't want to overwrite and we already have an entry
        if overwrite is True or data['position_exact'] is None:

            point = f"ST_GeomFromText('POINT({x_exact} {y_exact})',3031)"

            sql_string = f"UPDATE images_extracted SET " \
                         f"position_exact={point}, " \
                         f"footprint_exact_type='{footprint_type}' " \
                         f"WHERE image_id='{image_id}'"
            ctd.edit_data_in_db(sql_string, catch=catch, verbose=verbose, pbar=pbar)

    if debug_save_exact_photocenters_shape:
        shape_path = fld_shapefiles + "photocenters.shp"

        # Create a GeoDataFrame with the photocenter
        photocenter_gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy([x_exact], [y_exact]))
        photocenter_gdf.crs = "EPSG:3031"
        photocenter_gdf['image_id'] = image_id

        if shp_data is not None:
            photocenter_gdf['nr_points'] = shp_data["nr_points"]
            photocenter_gdf['avg_conf'] = shp_data["avg_conf"]
            photocenter_gdf['sat_min_x'] = shp_data['sat_min_x']
            photocenter_gdf['sat_max_x'] = shp_data['sat_max_x']
            photocenter_gdf['sat_min_y'] = shp_data['sat_min_y']
            photocenter_gdf['sat_max_y'] = shp_data['sat_max_y']

        # Create the shapefile if it doesn't exist
        if not os.path.exists(shape_path):
            photocenter_gdf.to_file(shape_path)
        else:
            # Read the existing shapefile
            existing_gdf = gpd.read_file(shape_path)

            # Check if the image_id is already in our existing photocenters
            if image_id in existing_gdf['image_id'].values:
                # which columns should be updated
                columns_to_update = ["geometry"]
                if shp_data is not None:
                    columns_to_update = columns_to_update + ["nr_points", "avg_conf",
                                                             "sat_min_x", "sat_max_x",
                                                             "sat_min_y", "sat_max_y"]

                filter_condition = existing_gdf['image_id'] == image_id

                existing_gdf.loc[filter_condition, columns_to_update] = photocenter_gdf.iloc[0][columns_to_update]

            else:
                # Append the new photocenter to the existing GeoDataFrame
                existing_gdf = pd.concat([existing_gdf, photocenter_gdf], ignore_index=True)

            # Save the updated GeoDataFrame to the shapefile
            existing_gdf.to_file(shape_path)

    # load the approx position from the database
    sql_string = f"SELECT x_coords, y_coords FROM images WHERE image_id='{image_id}'"
    data_approx = ctd.get_data_from_db(sql_string, catch=catch, verbose=verbose, pbar=pbar)
    x_approx = data_approx.iloc[0]['x_coords']
    y_approx = data_approx.iloc[0]['y_coords']

    # calculate the error vector between approx position and real position
    error_x = x_exact - x_approx
    error_y = y_exact - y_approx

    # store error in db
    if debug_save_error_vector_db:

        # don't save if we don't want to overwrite, and we already have an entry
        if overwrite is True or data['position_error_vector'] is None:

            sql_string = f"UPDATE images_extracted SET " \
                         f"position_error_vector='{str(error_x)};{str(error_y)}' " \
                         f"WHERE image_id='{image_id}'"
            ctd.edit_data_in_db(sql_string, catch=catch, verbose=verbose, pbar=pbar)

    if debug_save_error_vector_shape:
        shape_path = fld_shapefiles + "error_vectors.shp"

        # Create a GeoDataFrame with the line
        line = LineString([(x_exact, y_exact), (x_approx, y_approx)])
        line_gdf = gpd.GeoDataFrame(geometry=[line])
        line_gdf.crs = "EPSG:3031"
        line_gdf['image_id'] = image_id

        # Create the shapefile if it doesn't exist
        if not os.path.exists(shape_path):
            line_gdf.to_file(shape_path)
        else:
            # Read the existing shapefile
            existing_gdf = gpd.read_file(shape_path)

            # Check if the image_id is already in our existing lines
            if image_id in existing_gdf['image_id'].values:
                # Update the entry in existing_gdf
                existing_gdf.loc[existing_gdf['image_id'] == image_id, "geometry"] = line_gdf.iloc[0]['geometry']
            else:
                # Append the new line to the existing GeoDataFrame
                existing_gdf = pd.concat([existing_gdf, line_gdf], ignore_index=True)

            # Save the updated GeoDataFrame to the shapefile
            existing_gdf.to_file(shape_path)

    p.print_v(f"{image_id} saved in db", verbose=verbose, pbar=pbar)