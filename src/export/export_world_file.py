
def export_world_file(trans_matrix, output_path):
    """
    Export a world file from a transformation matrix.

    Parameters:
    - trans_matrix: np.ndarray. Transformation matrix as a numpy array.
    - output_path: str. The path to save the world file.

    The transformation matrix is expected to be in the form:
    | a, b, c |
    | d, e, f |
    | 0, 0, 1 |

    Where:
    a: pixel size in the x-direction
    b: rotation about the y-axis
    c: x-coordinate of the upper left pixel
    d: rotation about the x-axis
    e: pixel size in the y-direction (typically negative)
    f: y-coordinate of the upper left pixel
    """
    if trans_matrix.shape == (3, 3):
        # Assuming a full 3x3 matrix, we ignore the last row.
        trans_flat = trans_matrix.flatten()[:6]
    elif trans_matrix.shape == (6,):
        trans_flat = trans_matrix
    else:
        raise ValueError("Transformation matrix must be either 3x3 or a flat array of size 6.")

    # The elements of the transformation matrix mapping to the world file format.
    world_file_content = [
        trans_flat[0],  # pixel size in the x-direction
        trans_flat[1],  # rotation about the y-axis
        trans_flat[3],  # rotation about the x-axis
        trans_flat[4],  # negative of the pixel size in the y-direction
        trans_flat[2],  # x-coordinate of the center of the upper left pixel
        trans_flat[5],  # y-coordinate of the center of the upper left pixel
    ]

    # Writing to the world file
    with open(output_path, 'w') as file:
        for item in world_file_content:
            file.write(f"{item}\n")
