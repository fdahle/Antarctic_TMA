import numpy as np

from shapely.affinity import translate
from shapely.geometry import LineString, Point, Polygon

# import georef snippet functions
import src.georef.snippets.calc_transform as ct

debug_show_image_footprint = False

class GeorefCalc:

    def __init__(self, transform_method="rasterio", transform_order=3,):

        self.min_nr_of_images = 2
        self.max_range = 4

        self.transform_method = transform_method
        self.transform_order = transform_order

    def georeference(self, image, image_id, georeferenced_ids, georeferenced_footprints):

        # check if there are enough geo-referenced images to calculate a position
        if len(georeferenced_ids) < self.min_nr_of_images:
            return None, None, None, None

        # check if the image is already geo-referenced -> no calculation needed
        if image_id in georeferenced_ids:
            pass

        # Get the centroids of the footprints
        georeferenced_centers = [footprint.centroid for footprint in georeferenced_footprints]

        # derive the position of the image
        image_position = self._derive_image_position(image_id, georeferenced_ids, georeferenced_centers)

        image_footprint = self._derive_image_geometry(image_id, image_position,
                                                      georeferenced_ids, georeferenced_footprints)

        if debug_show_image_footprint:
            style_config = {
                'labels': [georeferenced_ids, None, image_id, None],
                'colors': ['blue', 'blue', 'red', 'red'],
            }

            import src.display.display_shapes as ds
            ds.display_shapes([georeferenced_centers, georeferenced_footprints,
                               image_position, image_footprint],
                              style_config=style_config, normalize=True)

        # get corners from footprint excluding the repeated last point
        tps_abs = np.asarray(list(image_footprint.exterior.coords)[:-1])

        # create tps for the image and sort them
        tps_img = np.array([[0, 0], [image.shape[1], 0], [image.shape[1], image.shape[0]], [0, image.shape[0]]])

        # create merged tie-point array
        tps = np.concatenate((tps_abs, tps_img), axis=1)

        # confidence is always 1
        conf = np.ones(tps.shape[0])

        transform, residuals = ct.calc_transform(image, tps,
                                                 transform_method=self.transform_method,
                                                 gdal_order=self.transform_order)

        return transform, residuals, tps, conf


    def _derive_image_position(self, image_id, georeferenced_ids, georeferenced_centers):

        # convert image ids to just their numbers
        image_nr = int(image_id[-4:])
        georeferenced_numbers = [int(item[-4:]) for item in georeferenced_ids]

        # Ensure the centers are sorted according to their numerical IDs
        sorted_indices = np.argsort(georeferenced_numbers)
        georeferenced_centers_sorted = [georeferenced_centers[i] for i in sorted_indices]
        georeferenced_numbers_sorted = [georeferenced_numbers[i] for i in sorted_indices]

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
            spatial_distance = Point(georeferenced_centers_sorted[i]).distance(Point(georeferenced_centers_sorted[i+1]))
            id_difference = abs(georeferenced_numbers_sorted[i+1] - georeferenced_numbers_sorted[i])
            adjusted_distance = spatial_distance / max(1, id_difference)  # Avoid division by zero
            distances_adjusted.append(adjusted_distance)

        # get an average distance for the adjusted distances
        avg_distance_adjusted = np.mean(distances_adjusted)

        # Find the position where the new image fits in
        position_index = next((i for i, num in enumerate(georeferenced_numbers_sorted) if num > image_nr), len(georeferenced_numbers_sorted))

        if position_index == 0:
            reference_point = georeferenced_centers_sorted[0]
            next_point = georeferenced_centers_sorted[1]
        elif position_index == len(georeferenced_centers_sorted):
            reference_point = georeferenced_centers_sorted[-1]
            prev_point = georeferenced_centers_sorted[-2]
        else:
            reference_point = georeferenced_centers_sorted[position_index - 1]
            next_point = georeferenced_centers_sorted[position_index]

        # Convert Shapely Points to numpy arrays for calculation
        reference_point_coords = np.array([reference_point.x, reference_point.y])
        if position_index == 0:
            next_point_coords = np.array([next_point.x, next_point.y])
            direction = next_point_coords - reference_point_coords
        elif position_index == len(georeferenced_centers_sorted):
            prev_point_coords = np.array([prev_point.x, prev_point.y])
            direction = reference_point_coords - prev_point_coords
        else:
            next_point_coords = np.array([next_point.x, next_point.y])
            direction = next_point_coords - reference_point_coords

        direction_normalized = direction / np.linalg.norm(direction)

        # Calculate the distance to adjust from the reference point
        if position_index == 0 or position_index == len(georeferenced_centers_sorted):
            distance_adjustment = avg_distance_adjusted
        else:
            id_difference = abs(image_nr - georeferenced_numbers_sorted[position_index - 1])
            distance_adjustment = avg_distance_adjusted * id_difference

        # Calculate the new position along the direction
        new_coords = reference_point_coords + direction_normalized * distance_adjustment
        new_x, new_y = new_coords

        # Adjust the y coordinate to ensure it is on the fitted line
        new_y = line_func(new_x)

        # create shapely point
        calculated_point = Point(new_x, new_y)

        return calculated_point

    def _derive_image_geometry(self, image_id, image_center, georeferenced_ids, georeferenced_footprints):

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
