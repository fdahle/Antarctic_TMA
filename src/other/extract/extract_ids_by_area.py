"""extract image ids from database based on area of interest"""

# Library imports
import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from shapely.wkt import loads as load_wkt
from typing import List, Union, Optional

# Local imports
import src.base.connect_to_database as ctd
import src.display.display_shapes as ds

# Debug flags
debug_show_footprints = False


def extract_ids_by_area(aoi: Union[Polygon, List[float]],
                        image_positions: Optional[gpd.GeoDataFrame] = None,
                        image_directions: Optional[List[str]] = None,
                        footprint_type: str = "exact",
                        min_percentage: int = 80,
                        complete_flightpaths: bool = False,
                        check_clusters: bool = False,
                        crs: int = 3031) -> List[str]:
    """
    Extracts image IDs from the database based on the area of interest and other criteria.

    Args:
        aoi: Area of interest as a Polygon or a bounding box list [min_x, min_y, max_x, max_y].
        image_positions: Optional GeoDataFrame containing image positions. If None,
            fetched from the database.
        image_directions: List of image directions to filter by. Defaults to ['L', 'V', 'R'].
        footprint_type: Type of footprint to consider. Can be "exact" or "approx". Defaults to "exact".
        min_percentage: Minimum percentage of overlap with AOI for an image to be considered. Defaults to 80.
        complete_flightpaths: Boolean indicating if complete flight paths should be considered.
            Defaults to False.
        check_clusters: Boolean indicating if the function should check for multiple clusters of footprints.
            Defaults to False.
        crs: Coordinate reference system to be used for GeoDataFrames. Defaults to EPSG:3031.

    Returns:
        List of image IDs that match the criteria.

    """

    print("Extract ids")

    # set default value for image_directions
    if image_directions is None:
        image_directions = ['L', 'V', 'R']

    # establish connection to psql
    conn = ctd.establish_connection()

    # if we don't have image positions yet, we can get them from the database
    if image_positions is None:

        # define the colum type for the footprint
        if footprint_type == "exact":
            footprint_column = "footprint_exact"
        elif footprint_type == "approx":
            footprint_column = "footprint_approx"
        else:
            raise ValueError("footprint_type must be either 'exact' or 'approx'")

        # get the image positions as wkt points from the database
        sql_string = f"SELECT image_id, ST_AsText({footprint_column}) as position" \
                     " FROM images_extracted"
        data = ctd.execute_sql(sql_string, conn)

        # Convert the WKT strings to Shapely geometries
        geometries = data['position'].apply(load_wkt)

        # Create a GeoDataFrame from these geometries
        image_positions = gpd.GeoDataFrame(data, geometry=geometries, crs=crs)

    # check if aoi is already a polygon
    if isinstance(aoi, Polygon):
        poly = aoi
    elif isinstance(aoi, str):
        poly = load_wkt(aoi)
    else:
        # convert BoundingBox to polygon
        poly = Polygon([(aoi[0], aoi[1]), (aoi[0], aoi[3]),
                        (aoi[2], aoi[3]), (aoi[2], aoi[1]),
                        (aoi[0], aoi[1])
                        ])

    # convert polygon to geopandas
    poly_gpd = gpd.GeoDataFrame(geometry=[poly], crs=crs)  # noqa

    # filter the points that are intersect this polygon
    filtered_shape_data = gpd.sjoin(image_positions, poly_gpd, predicate="intersects")

    # Calculate the overlap area with AOI in percentage of footprint
    filtered_shape_data['total_area'] = filtered_shape_data['geometry'].area
    filtered_shape_data['overlap_area'] = filtered_shape_data['geometry'].intersection(poly).area
    filtered_shape_data['overlap_percentage'] = (filtered_shape_data['overlap_area'] / filtered_shape_data[
        'total_area']) * 100

    # Filter the image positions based on the minimum percentage of overlap
    filtered_shape_data = filtered_shape_data[filtered_shape_data['overlap_percentage'] > min_percentage]

    # get all images from the flightpaths that intersect the aoi
    if complete_flightpaths:

        # get the TMA numbers from the image_ids
        if 'image_id' in filtered_shape_data.columns:
            filtered_shape_data['TMA_num'] = filtered_shape_data['image_id'].str[2:6]

        # ensure the TMA numbers follow the correct format
        filtered_shape_data['TMA_num'] = filtered_shape_data['TMA_num'].astype(str).str.zfill(4)

        # get the unique values for TMA numbers
        flight_paths = filtered_shape_data['TMA_num'].unique()
        flight_paths_tuple = tuple(flight_paths)

        # get all ids from database with these TMA numbers
        sql_string = f"SELECT image_id FROM images WHERE tma_number IN {flight_paths_tuple}"  # noqa
        complete_data = ctd.execute_sql(sql_string, conn)

    else:
        complete_data = pd.DataFrame(filtered_shape_data['image_id'], columns=['image_id'])

    # init list for all ids
    ids = []

    # Filter and collect image IDs based on specified image directions
    for direction in image_directions:
        if direction in ['L', 'V', 'R']:
            images_direction = complete_data[complete_data['image_id'].str.contains(direction)]['image_id']
            ids += images_direction.tolist()

    # show the footprints
    if debug_show_footprints:
        # get the geometries of the ids from filtered_shape_data
        footprints = filtered_shape_data[filtered_shape_data['image_id'].isin(ids)]['geometry']

        # convert aoi to shapely polygon
        aoi_poly = poly_gpd['geometry'].iloc[0]

        # convert to list of shapely polygons
        footprints = [geom for geom in footprints if isinstance(geom, Polygon)]

        style_config = {
            "colors": ["red", "blue"],
        }
        ds.display_shapes([aoi_poly, footprints], normalize=False, style_config=style_config)

    if check_clusters:
        # Combine all footprints into a single unary union to identify connected components
        combined_footprints = unary_union(filtered_shape_data['geometry'].tolist())

        if isinstance(combined_footprints, MultiPolygon):
            raise ValueError(f"Multiple clusters found in footprints.")

    print(f"{len(ids)} ids are extracted")

    return ids
