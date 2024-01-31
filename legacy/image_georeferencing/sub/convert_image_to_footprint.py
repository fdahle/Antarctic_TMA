import numpy as np
import rasterio.features
import shapely.ops

from shapely.geometry import Polygon, MultiPolygon

import base.print_v as p


def convert_image_to_footprint(img, image_id, transform, catch=True, verbose=False, pbar=None):
    """
    convert_image_to_footprint(img, image_id, transform, catch, verbose, pbar):
    Takes a geo-referenced image and creates a shapely polygon based on the approx_footprint of the image.
    Args:
        img (np-array): The image we want to convert to an approx_footprint
        image_id (String): The image_id of the image we want to convert
        transform (transform): A transform obj from rasterio. Required to geo-reference the approx_footprint
        catch (Boolean): If true, we catch every error that is happening and return instead None
        verbose (Boolean): If true, we print information that is happening during execution of the function
        pbar (Tqdm-progressbar): If this is not None, the printing will be added to a tqdm-progress bar
    Returns:
        polygon (Shapely-polygon): A polygon with the same extent as the polygon
    """

    p.print_v(f"Start: convert_image_to_footprint ({image_id})", verbose=verbose, pbar=pbar)

    try:
        # create a mask of this image
        mask = np.ones_like(img)
        mask[img == 0] = 0

        # convert the raster cell to polygons
        shapes = rasterio.features.shapes(mask, transform=transform)

        # merge all polygons
        shape = shapely.ops.unary_union([shapely.geometry.shape(shape) for shape, val in shapes if val == 1])

        # flatten the polygon lines
        polygon = shape.simplify(100)

        # Initialize the biggest polygon and its area
        final_poly = None
        max_area = 0

        if isinstance(polygon, Polygon):
            polygons = [polygon]  # Convert single polygon to a list of polygons
        elif isinstance(polygon, MultiPolygon):
            polygons = polygon.geoms  # Access the individual polygons within the MultiPolygon
        else:
            raise ValueError("The type of polygon is undefined")

        # Iterate over all polygons
        for poly in polygons:

            # Calculate the area of the current polygon
            area = poly.area

            # If the current polygon's area is larger than the current maximum
            if area > max_area:
                # Update the biggest polygon and the maximum area
                final_poly = poly
                max_area = area

        # Create a new polygon without any interior rings
        final_poly = Polygon(final_poly.exterior)

    except (Exception,) as e:
        if catch:
            p.print_v(f"Failed: convert_image_to_footprint ({image_id})", verbose=verbose, pbar=pbar)
            return None
        else:
            raise e

    p.print_v(f"Finished: convert_image_to_footprint ({image_id})", verbose=verbose, pbar=pbar)

    return final_poly


if __name__ == "__main__":

    img_id = "CA184832V0114"
    _img_path = f"/data_1/ATM/data_1/playground/georef4/tiffs/sat/{img_id}.tif"

    import base.load_image_from_file as liff
    _img, _transform = liff.load_image_from_file(_img_path, return_transform=True)

    _poly = convert_image_to_footprint(_img, img_id, _transform)

    print(_poly)
