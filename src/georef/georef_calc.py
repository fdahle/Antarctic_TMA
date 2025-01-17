"""georeference images by calculating the footprint"""
import copy

# Library imports
import numpy as np
import shapely
from shapely.affinity import translate
from shapely.geometry import Point, Polygon
from typing import Optional, Tuple

# Local imports
import src.georef.snippets.calc_transform as ct

# debug plots
debug_display_image_footprint = True
print_debug = True


class GeorefCalc:
    """
    A class for geo-referencing historical images by calculating their footprints.
    """

    def __init__(self, min_nr_of_images: int = 3, max_range: int = 4,
                 transform_method: str = "rasterio", transform_order: int = 3):
        """
        Initialize the GeoRefCalc class with various settings for geo-referencing historical images
        by calculating the footprint of the image derived by already geo-referenced images from the
        same flight path.

        Args:
            min_nr_of_images (int): The minimum number of images needed to calculate a position.
                Defaults to 3.
            max_range (int): The maximum distance of neighbouring images that are used to calculate
                the average footprint.
            transform_method (str): The method used for the transformation calculation.
                Defaults to "rasterio".
            transform_order (int): The order of transformation for the geo-referencing process.
                Defaults to 3.
        """
        self.min_nr_of_images = min_nr_of_images
        self.max_range = max_range

        self.transform_method = transform_method
        self.transform_order = transform_order

    def georeference(self, image: np.ndarray, image_id: str,
                     georeferenced_ids: list[str],
                     georeferenced_footprints: list[shapely.Polygon]) -> Tuple[Optional[np.ndarray],
                                                                               Optional[np.ndarray],
                                                                               Optional[np.ndarray],
                                                                               Optional[np.ndarray]]:
        """
        Geo-reference an image using a list of neighbouring geo-referenced images and their footprints.
        Args:
            image (np.ndarray): The input image to be georeferenced.
            image_id (str): The image id of the input image.
            georeferenced_ids (list[str]): A list of image ids of the geo-referenced images.
            georeferenced_footprints (list[shapely.Polygon]): A list of footprints of the
                geo-referenced images.
        Returns:
            transform (np.ndarray): The transformation matrix of the geo-referenced image as a NumPy array.
            residuals(np.ndarray): The residuals of the transformation points as a NumPy array.
            tps (np.ndarray): The tie points between the input image and the satellite image as a NumPy array.
            conf (np.ndarray): The confidence scores associated with the tie points as a NumPy array.
        """

        # Filter out geo-referenced images and footprints that don't match the flight path
        filtered_georef_ids = []
        filtered_footprints= []
        for i, geo_id in enumerate(georeferenced_ids):
            if geo_id[2:6] == image_id[2:6]:
                filtered_georef_ids.append(geo_id)
                filtered_footprints.append(georeferenced_footprints[i])

        # check if there are enough geo-referenced images to calculate a position
        if len(filtered_georef_ids) < self.min_nr_of_images:
            print("Not enough images to calculate a position")
            return None, None, None, None

        # Get the centroids of the footprints
        filtered_centers = [footprint.centroid for footprint in filtered_footprints]

        # check if the image is already geo-referenced -> no calculation needed
        if image_id in filtered_georef_ids:

            # get the index of the image in the list
            index = filtered_georef_ids.index(image_id)

            # get the equivalent position & footprint
            image_position = filtered_centers[index]
            image_footprint = filtered_footprints[index]

        # calculate image position and footprint
        else:
            # derive the position of the image
            image_position = self._derive_image_position(image_id, filtered_georef_ids,
                                                         filtered_centers)

            image_footprint = self._derive_image_geometry(image_id, image_position,
                                                          filtered_georef_ids,
                                                          filtered_footprints)

        if debug_display_image_footprint:
            style_config = {
                'labels': [filtered_georef_ids, None, image_id, None],
                'colors': ['blue', 'blue', 'red', 'red'],
            }

            import src.display.display_shapes as ds
            ds.display_shapes([filtered_centers, filtered_footprints,
                               image_position, image_footprint],
                              style_config=style_config, normalize=True,
                              save_path=f"/home/fdahle/Desktop/georef_test/{image_id}.png")

        # get corners from footprint excluding the repeated last point
        tps_abs = np.asarray(list(image_footprint.exterior.coords)[:-1])

        # create tps for the image and sort them
        tps_img = np.array([[0, 0], [image.shape[1], 0], [image.shape[1], image.shape[0]], [0, image.shape[0]]])

        # create merged tie-point array
        tps = np.concatenate((tps_abs, tps_img), axis=1)

        # confidence is always 1
        conf = np.ones(tps.shape[0])

        # calculate the transformation-matrix
        transform, residuals = ct.calc_transform(image, tps,
                                                 transform_method=self.transform_method,
                                                 gdal_order=self.transform_order)

        return transform, residuals, tps, conf

    @staticmethod
    def _derive_image_position(image_id: str, georeferenced_ids: list[str],
                               georeferenced_centers: list[Point]) -> Point:
        """
        This function calculates the position(=Point) where a new image should be placed based on
        the spatial arrangement and numerical ID order of existing georeferenced images.
        It fits a linear model to the positions of georeferenced images, calculates the average
        adjusted distance between them, and places the new image accordingly on the line.
        Args:
            image_id (str): The ID of the new image to be positioned.
            georeferenced_ids (list[str]): A list of IDs for the already georeferenced images.
            georeferenced_centers (list[Point]): A list of Shapely Point objects representing
                the centers of the georeferenced images.
        Returns:
            Point: A Shapely Point object representing the calculated position/center for the
                new image.
        """

        if print_debug:
            print("Georeferenced ids:")
            print(georeferenced_ids)

        # convert image ids to just their numbers
        image_nr = int(image_id[-4:])
        georeferenced_numbers = [int(item[-4:]) for item in georeferenced_ids]

        # Ensure the centers are sorted according to their numerical IDs
        sorted_indices = np.argsort(georeferenced_numbers)
        georeferenced_centers_sorted = [georeferenced_centers[i] for i in sorted_indices]
        georeferenced_numbers_sorted = [georeferenced_numbers[i] for i in sorted_indices]

        if print_debug:
            print("Image_nr:", image_nr)
            print("Georef_numbers:", georeferenced_numbers_sorted)

        # Extract x and y coordinates for line fitting
        x = [point.x for point in georeferenced_centers]
        y = [point.y for point in georeferenced_centers]

        # Fit a line to these centers using polyfit for a degree 1 polynomial (linear fit)
        line_fit = np.polyfit(x, y, 1)

        # Create a line function from the fit
        line_func = np.poly1d(line_fit)

        # Calculate adjusted distances between consecutive points, considering their numerical IDs
        distances_adjusted = []
        for i in range(len(georeferenced_centers_sorted) - 1):
            spatial_distance = Point(georeferenced_centers_sorted[i]).distance(
                Point(georeferenced_centers_sorted[i + 1]))
            id_difference = abs(georeferenced_numbers_sorted[i + 1] - georeferenced_numbers_sorted[i])
            adjusted_distance = spatial_distance / max(1, id_difference)  # Avoid division by zero
            distances_adjusted.append(adjusted_distance)

        if print_debug:
            print("distances_adjusted:")
            print(distances_adjusted)

        # get an average distance for the adjusted distances
        avg_distance_adjusted = np.mean(distances_adjusted)

        if print_debug:
            print("Average distance adjusted:", avg_distance_adjusted)

        # Find the position where the new image fits in
        position_index = next((i for i, num in enumerate(georeferenced_numbers_sorted) if num > image_nr),
                              len(georeferenced_numbers_sorted))

        # init points to satisfy the linter
        next_point = prev_point = None
        next_id = prev_id = None

        if print_debug:
            print("Position index:", position_index)

        # calculate the reference point and the next point
        if position_index == 0:
            next_point = georeferenced_centers_sorted[1]
            next_id = georeferenced_numbers_sorted[1]
        elif position_index == len(georeferenced_centers_sorted):
            prev_point = georeferenced_centers_sorted[-2]
            prev_id = georeferenced_numbers_sorted[-2]
        else:
            prev_point = georeferenced_centers_sorted[position_index - 1]
            prev_id = georeferenced_numbers_sorted[position_index - 1]
            next_point = georeferenced_centers_sorted[position_index]
            next_id = georeferenced_numbers_sorted[position_index]

        if print_debug:
            print("Prev point:", prev_point, "ID:", prev_id)
            print("Next point:", next_point, "ID:", next_id)

        # Convert Shapely Points to numpy arrays for calculation
        if position_index == 0:
            if print_debug:
                print("pos at 0")
            ref_coords = np.array([next_point.x, next_point.y])
            ref_id = next_id
            direction = -1
        elif position_index == len(georeferenced_centers_sorted):
            if print_debug:
                print("pos at end")
            ref_coords = np.array([prev_point.x, prev_point.y])
            ref_id = prev_id
            direction = 1
        else:
            if print_debug:
                print("pos at between")
            ref_coords = np.array([prev_point.x, prev_point.y])
            ref_id = prev_id
            direction = 1

        if print_debug:
            print("ref id", ref_id)

        # sometimes we need to switch the direction
        if x[0] > x[-1]:
            if print_debug:
                print("x0 bigger x1")
            direction = -direction
        else:
            if print_debug:
                print("x0 smaller x1")
            direction = direction

        if print_debug:
            print("direction", direction)

        diff_ids = abs(image_nr - ref_id)

        if print_debug:
            print("diff ids: ", diff_ids)

        distance_adjustment = avg_distance_adjusted * diff_ids

        # Calculate the new position along the direction
        new_coords = ref_coords + distance_adjustment * direction
        new_x, new_y = new_coords

        if print_debug:
            print("Distance adjustment", distance_adjustment)
            print("New Coords:", new_coords)

        # Adjust the y coordinate to ensure it is on the fitted line
        new_y = line_func(new_x)

        # create shapely point
        calculated_point = Point(new_x, new_y)

        return calculated_point

    def _derive_image_geometry(self, image_id: str, image_center: Point, georeferenced_ids: list[str],
                               georeferenced_footprints: list[Polygon]) -> Polygon:
        """
        This method derives a footprint polygon for a new image by calculating the weighted
        average position of the corners of nearby georeferenced footprints. The weights are based
        on the numerical distance between the new image's ID and the IDs of georeferenced images,
        favoring closer numerical IDs to influence the derived geometry more strongly.
        Args:
            image_id (str): The ID of the new image, used to calculate numerical distances to
                georeferenced images.
            image_center (Point): A Shapely Point object representing the center of the new image.
            georeferenced_ids (List[str]): A list of IDs for georeferenced images.
            georeferenced_footprints (List[Polygon]): A list of Shapely Polygon objects representing
                the footprints of georeferenced images.
        Returns:
            Polygon: A Shapely Polygon object representing the derived footprint of the new image,
                adjusted to be centered on the provided image center.
        """

        # convert image ids to just their numbers
        image_nr = int(image_id[-4:])
        georeferenced_numbers = [int(item[-4:]) for item in georeferenced_ids]

        # Convert georeferenced footprints to numpy arrays for easier manipulation
        georeferenced_coords = [np.array(footprint.exterior.coords)[:-1] for footprint in georeferenced_footprints]

        # Calculate numerical distances between image_nr and georeferenced_numbers
        numerical_distances = [abs(image_nr - num) for num in georeferenced_numbers]
        # Convert numerical distances to weights (inverse relationship)
        weights = [1 / (distance + 1) for distance in numerical_distances]  # Add 1 to avoid division by zero

        # Calculate weighted offsets of each footprint's center from the image center
        offsets = [np.mean(coords, axis=0) - np.array([image_center.x, image_center.y]) for coords in
                   georeferenced_coords]

        # Sort georeferenced footprints by their distance to the image center's x-coordinate, including weights
        sorted_footprints_by_distance = sorted(zip(georeferenced_coords, offsets, weights),
                                               key=lambda item: abs(np.mean(item[0][:, 0]) - image_center.x))

        # Select the closest X footprints in both directions, including their offsets and weights
        max_range = min(self.max_range, len(sorted_footprints_by_distance))
        closest_footprints_with_offsets_weights = sorted_footprints_by_distance[:max_range]

        # Calculate the weighted average positions of the corners and the weighted average offset
        avg_corners = np.average([coords for coords, _, _ in closest_footprints_with_offsets_weights],
                                 axis=0, weights=[weight for _, _, weight in closest_footprints_with_offsets_weights])
        avg_offset = np.average([offset for _, offset, _ in closest_footprints_with_offsets_weights],
                                axis=0, weights=[weight for _, _, weight in closest_footprints_with_offsets_weights])

        # Adjust the average corners by the average offset to position the derived footprint correctly
        adjusted_avg_corners = avg_corners + avg_offset

        # Create the average footprint polygon, ensuring to close the polygon by repeating the first point
        derived_footprint = Polygon(np.vstack([adjusted_avg_corners, adjusted_avg_corners[0]]))

        # Calculate the translation required to move the derived footprint's centroid to the image center
        centroid = derived_footprint.centroid
        dx = image_center.x - centroid.x
        dy = image_center.y - centroid.y

        # Translate the derived footprint to the image center
        translated_footprint = translate(derived_footprint, xoff=dx, yoff=dy)

        return translated_footprint
