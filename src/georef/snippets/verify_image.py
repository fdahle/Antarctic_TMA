import numpy as np

def verify_image(image, transform):

    # Get the pixel coordinates of the four corners of the image
    rows, cols = image.shape[0], image.shape[1]
    corners = np.array([(0, 0), (0, rows), (cols, rows), (cols, 0)])

    # Convert to abs coordinates
    corners_abs = [transform * xy for xy in corners]

    # Calculate distances between corner points to get all side lengths
    distances = [np.linalg.norm(corners_abs[i] - corners_abs[(i + 1) % 4]) for i in range(4)]



    # get difference between height and width of image
    diff = np.abs(((distances[0] - distances[1]) / distances[1]) * 100)

    print(diff)