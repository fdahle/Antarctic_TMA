import numpy as np
from PIL import Image
from typing import Tuple

import base.print_v as p


def rotate_image(image, angle, expand=True, start_angle=0,
                 return_corners=False,
                 catch=True, verbose=False, pbar=None):
    """
    rotate_image(image, angle, expand, catch, verbose, pbar):
    This function rotates an input image with the given number of degrees.
    Args:
        image (np-array): The image that should be rotated
        angle (integer): How much should the image be rotated clockwise.
        expand (bool): If true, the rotated image can be larger than the original,
            otherwise the image will be cropped to the original size
        start_angle (int): Some images start rotation not at 0 degree, so we must account for that
        catch (Boolean, True): If true and something is going wrong (for example no fid points),
            the operation will continue and not crash
        verbose (Boolean, False): If true, the status of the operations are printed
        pbar (tqdm-progress-bar): If this is true, the text output will be not shown as text, but
            as a description in a tqdm-progress-bar
    Returns:
        img (np-array): The rotated image
    """

    p.print_v("Start: rotate_image", verbose=verbose, pbar=pbar)

    def transform_pixel_coordinates(
            pixel_coordinates: Tuple[int, int],
            angle: float,
            image: Image,
            rotated_image: Image,
    ) -> Tuple[int, int]:
        """
        Transform pixel coordinates.

        Parameters
        ----------
        pixel_coordinates : Tuple[int, int]
        angle : float
        image : Image
        rotated_image : Image

        Returns
        -------
        Tuple[int, int]

        """

        x, y = pixel_coordinates

        center = (image.shape[1] / 2, image.shape[0] / 2)
        transformed_center = (rotated_image.shape[1] / 2, rotated_image.shape[0] / 2)

        angle_radians = -np.deg2rad(angle)

        x -= center[0]
        y -= center[1]

        x_transformed = x * np.cos(angle_radians) - y * np.sin(angle_radians)
        y_transformed = x * np.sin(angle_radians) + y * np.cos(angle_radians)

        x_transformed += transformed_center[0]
        y_transformed += transformed_center[1]

        return int(x_transformed), int(y_transformed)

    rot_angle = 360 - angle + 90 - start_angle

    try:
        img = Image.fromarray(image)
        img = img.rotate(rot_angle, expand=expand)
        img = np.asarray(img)  # noqa

        if return_corners:

            corners = [(0, 0), (image.shape[1], 0), (0, image.shape[0]), (image.shape[1], image.shape[0])]
            corners_rotated = []
            for corner in corners:
                test_x, test_y = transform_pixel_coordinates(corner, rot_angle, image, img)
                corners_rotated.append((test_x, test_y))

    except (Exception,) as e:
        if catch:
            p.print_v("Failed: rotate_image", verbose=verbose, pbar=pbar)
            return None
        else:
            raise e

    p.print_v("Finished: rotate_image", verbose=verbose, pbar=pbar)

    if return_corners:
        return img, corners_rotated
    else:
        return img


if __name__ == "__main__":
    image_id = "CA182632V0127"

    import load_image_from_file as liff

    _img = liff.load_image_from_file(image_id)

    import connect_to_db as ctd

    sql_string = f"SELECT * FROM images WHERE image_id='{image_id}'"
    data = ctd.get_data_from_db(sql_string)
    data = data.iloc[0]

    azimuth = data["azimuth"]

    _img = rotate_image(_img, azimuth)

    import display.display_images as di

    di.display_images(_img)
