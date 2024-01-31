

def calc_camera_position(polygon, catch=True):
    """
    calc_camera_position(polygon, catch)
    calculate the camera position from an approx_footprint based on a centroid
    Args:
        polygon (Shapely-polygon): The polygon for that we want to calculate the position
        catch (Boolean, True): If true and something is going wrong (for example no fid points),
            the operation will continue and not crash
    Returns:
        x (float): The x-coordinate of the calculated camera position
        y (float): The y-coordinate of the calculated camera position
    """
    try:
        # calculate the camera position (just by taking the centroid)
        centroid = polygon.centroid
        x, y = centroid.coords[0]

    except (Exception,) as e:
        if catch:
            return None, None
        else:
            raise e

    return x, y
