import numpy as np

from shapely.geometry import LineString


# import georef snippet functions
import src.georef.snippets.calc_transform as ct

class GeorefCalc:

    def __init__(self):

        self.min_nr_of_images = 2
        self.max_range = 4

    def georeference(self, image, image_id, georeferenced_ids, georeferenced_footprints):

        # remove all geo-referenced images that are not on the same line as the image
        # TODO: implement this

        # check if there are enough geo-referenced images to calculate a position
        if len(georeferenced_ids) < self.min_nr_of_images:
            return None, None, None, None


        transform, residuals = ct.calc_transform(image, tps,
                                                 transform_method=self.transform_method,
                                                 gdal_order=self.transform_order)

        return transform, residuals, tps, conf


def _derive_position(image_id, georeferenced_ids, georeferenced_footprints):

    # convert image ids to just their numbers
    image_nr = image_id[-4:]
    georeferenced_numbers = [item[-4:] for item in georeferenced_ids]

    # Get the centroids of the line footprints
    line_centers = [line.centroid for line in georeferenced_footprints]

    # Extract x and y coordinates for line fitting
    x = [point.x for point in line_centers]
    y = [point.y for point in line_centers]

    # Fit a line to these centers using polyfit for a degree 1 polynomial (linear fit)
    line_fit = np.polyfit(x, y, 1)

    # Create a line function from the fit
    line_func = np.poly1d(line_fit)

    # Generate a long line;
    x_long = np.linspace(min(x) - 1, max(x) + 1, num=2)
    y_long = line_func(x_long)

    # Create a shapely LineString from the fitted line
    fitted_line = LineString(zip(x_long, y_long))
