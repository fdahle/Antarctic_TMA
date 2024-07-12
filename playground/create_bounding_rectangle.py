from shapely.geometry import Polygon

def create_bounding_rectangle(footprints):
    """
    Create a bounding rectangle around the footprints

    :param footprints: list of shapely.Polygon objects
    :return: shapely.Polygon object
    """

    # Get the bounding box of the footprints
    min_x = min([footprint.bounds[0] for footprint in footprints])
    min_y = min([footprint.bounds[1] for footprint in footprints])
    max_x = max([footprint.bounds[2] for footprint in footprints])
    max_y = max([footprint.bounds[3] for footprint in footprints])

    # Create the bounding rectangle
    bounding_rectangle = Polygon([(min_x, min_y), (min_x, max_y), (max_x, max_y), (max_x, min_y)])

    return bounding_rectangle
