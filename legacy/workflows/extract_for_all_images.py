import json
import os
import subprocess

import shapely
from tqdm import tqdm

import image_extraction.calc_image_complexity as cic
import image_extraction.extract_altimeter as ea
import image_extraction.extract_height as eh
import image_extraction.extract_image_parameters as eip
import image_extraction.extract_text_tesseract as ett
import image_extraction.update_table_images_extracted as utie

import base.connect_to_db as ctd
import base.load_image_from_file as liff
import base.print_v as p
import base.remove_borders as rb
import base.update_failed_ids as ufi

debug_extract_height = False
debug_extract_text = True
debug_extract_params = True
debug_extract_complexity = True


def extract_for_all_images(image_ids,
                           extract_text_method=None,
                           complexity_highscore=None,
                           paddle_path=None, conda_path=None,
                           overwrite=False, catch=True, verbose=False):
    """
    extract_for_all_images(images_ids, catch, verbose, overwrite):
    This function extracts information from all the images. See the specific
     functions to see what is happening
    Args:
        image_ids (List): A list of image_ids
        extract_text_method (String): Are we extracting text with 'tesseract' or 'paddle'
        complexity_highscore (int): A number telling the maximum complexity of a highscore (see calc_image_complexity)
        paddle_path (String): When extracting with paddle, we need the path to the paddle python file
        conda_path (String): When extracting with paddle, we need the path to the conda env
        overwrite (Boolean): If true, all values and data will be overwritten with new values,
            if false, existing values and data will not be changed
        catch (Boolean): If true, we catch every error that is happening and return instead None
        verbose (Boolean): If true, we print information that is happening during execution of
            the function
    Returns:
        -
    """

    p.print_v(f"Start: extract_for_all_images ({len(image_ids)} images)", verbose=verbose)

    # load the json to get default values
    project_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    json_folder = project_folder + "/image_extraction"
    with open(json_folder + "/params.json") as j_file:
        json_data = json.load(j_file)

    if extract_text_method is None:
        extract_text_method = json_data["extract_text_method"]

    if complexity_highscore is None:
        complexity_highscore = json_data["complexity_highscore"]

    if paddle_path is None:
        paddle_path = json_data["path_file_paddle_python"]

    if conda_path is None:
        conda_path = json_data["path_file_paddle_conda"]

    # initialize the failed image_id manager
    fail_manager = ufi.FailManager("extract_for_all_images")

    # iterate all image_ids
    for image_id in (pbar := tqdm(image_ids)):

        p.print_v(f"Runtime: extract_for_all_images: {image_id}", verbose=verbose, pbar=pbar)

        # global var for image
        image = None

        # get already existing table data
        if overwrite is False:
            sql_string = f"SELECT * FROM images_extracted WHERE image_id='{image_id}'"
            table_data = ctd.get_data_from_db(sql_string, catch=catch, verbose=verbose, pbar=pbar)

            # somehow there's no row -> fail and continue
            if len(table_data) == 0 and catch:
                fail_manager.update_failed_id(image_id, "no row in table images_extracted")
                continue
            table_data = table_data.iloc[0]
        else:
            table_data = None

        if debug_extract_height:
            # check if we already have information for the altimeter
            if table_data is not None and \
                    table_data["altimeter_x"] is not None and \
                    table_data["altimeter_y"] is not None and \
                    table_data["altimeter_width"] is not None and \
                    table_data["altimeter_height"] is not None and \
                    table_data["circle_pos"] is not None:

                p.print_v(f"Runtime: extract_for_all_images (Altimeter and circle already extracted for {image_id})",
                          verbose=verbose, pbar=pbar)

                # create bounding box
                min_x = int(table_data["altimeter_x"])
                min_y = int(table_data["altimeter_y"])
                max_x = int(table_data["altimeter_x"]) + int(table_data["altimeter_width"])
                max_y = int(table_data["altimeter_y"]) + int(table_data["altimeter_height"])
                altimeter_bbox = [min_x, max_x, min_y, max_y]

                # create circle
                circle = table_data["circle_pos"]
                circle = circle.split(",")
                circle = [int(i) for i in circle]

            else:

                # check if we need to load the image
                if image is None:

                    # load the image
                    image = liff.load_image_from_file(image_id, catch=catch, verbose=verbose, pbar=pbar)

                    # check if we could load the image
                    if image is None:
                        fail_manager.update_failed_id(image_id, "load_image")
                        continue

                # get the position of the altimeter subset and the position of the circle in the subset
                # [min_x, max_x, min_y, max_y]
                altimeter_bbox, circle = ea.extract_altimeter(image, image_id,
                                                              catch=catch, verbose=verbose, pbar=pbar)

                # when an error happened during calculation
                if altimeter_bbox is None or circle is None:
                    fail_manager.update_failed_id(image_id, "extract_altimeter")

                # when no altimeter was found
                if altimeter_bbox:

                    # update the altimeter position
                    success = utie.update_table_images_extracted(image_id, "altimeter", altimeter_bbox,
                                                                 overwrite=overwrite,
                                                                 catch=catch, verbose=verbose, pbar=pbar)

                    # check if it was possible to update the table
                    if success is None:
                        fail_manager.update_failed_id(image_id, "update_table:altimeter")
                        continue

                if circle:
                    # update circle in table
                    success = utie.update_table_images_extracted(image_id, "circle", circle,
                                                                 overwrite=overwrite,
                                                                 catch=catch, verbose=verbose, pbar=pbar)

                    # check if it was possible to update the table
                    if success is None:
                        fail_manager.update_failed_id(image_id, "update_table:circle")
                        continue

            # check if we already have height information
            if table_data is not None and table_data["height"] is not None:
                p.print_v(f"Height is already extracted for {image_id}", verbose=verbose, pbar=pbar)
            else:

                # check if we need to load the image
                if image is None:

                    # load the image
                    image = liff.load_image_from_file(image_id, catch=catch, verbose=verbose, pbar=pbar)

                    # check if we could load the image
                    if image is None:
                        fail_manager.update_failed_id(image_id, "load_image")
                        continue

                # check if we have altimeter and circle
                if altimeter_bbox and circle:
                    # extract height
                    height = eh.extract_height(image, image_id, altimeter_bbox, circle,
                                               catch=catch, verbose=verbose, pbar=pbar)

                    if height is None:
                        fail_manager.update_failed_id(image_id, "extract_height")

                    # update only makes sense if we have found a height
                    if height is not None:

                        # update height in table
                        success = utie.update_table_images_extracted(image_id, "height", height,
                                                                     overwrite=overwrite,
                                                                     catch=catch, verbose=verbose, pbar=pbar)

                        # check if it was possible to update the table
                        if success is None:
                            fail_manager.update_failed_id(image_id, "update_table_images_extracted:height")
                            continue

        # init variable
        text = None

        if debug_extract_text:
            # check if we already have text information
            if table_data is not None and \
                    table_data["text_bbox"] is not None and \
                    table_data["text_content"] is not None and \
                    overwrite is False:
                p.print_v(f"Text is already extracted for {image_id}", verbose=verbose, pbar=pbar)

                # get the info from the table
                text = table_data["text_content"]

            else:

                # check if we need to load the image
                if image is None:

                    # load the image
                    image = liff.load_image_from_file(image_id, catch=catch, verbose=verbose, pbar=pbar)

                    # check if we could load the image
                    if image is None:
                        fail_manager.update_failed_id(image_id, "load_image_from_file")
                        continue

                # extract text position and content
                if extract_text_method == "tesseract":
                    text_nb, text_bounds_nb, _ = ett.extract_text_tesseract(image, image_id, binarize_image=True,
                                                                            catch=catch, verbose=verbose, pbar=pbar)
                    text_b, text_bounds_b, _ = ett.extract_text_tesseract(image, image_id, binarize_image=False,
                                                                          catch=catch, verbose=verbose, pbar=pbar)

                    text = text_nb + text_b
                    text_bounds = text_bounds_nb + text_bounds_b
                elif extract_text_method == "paddle":

                    file_path = os.path.abspath(paddle_path)

                    # all variables must be bytes or string
                    conda_env = conda_path.split("/")[-1]
                    conda_fld = conda_path[0:len(conda_path)-(len(conda_env)+1)]
                    catch_str = str(catch)

                    # the arguments for the subprocess
                    cmd = [conda_fld, "run", "-n", conda_env, "python", file_path,
                           "--image_id", image_id, "--binarize_image_str", str(False),
                           "--catch_str", catch_str, "--verbose_str", str(False)]

                    # call the subprocess
                    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

                    # parse the output
                    output = result.stdout.decode().strip()
                    output = output.split("\n")

                    # get text content
                    if len(output[0]) > 0:
                        text = output[0].replace("\x1b[0m", "")
                        if text == "[]":
                            text = None
                        else:
                            text = text[2:-2]
                            text = text.split("', '")
                    else:
                        text = None

                    # get text pos
                    if len(output[1]) > 0:
                        text_bounds = []
                        pos = output[1].replace("\x1b[0m", "")
                        if pos == "[]":
                            text_bounds = None
                        else:
                            pos = pos[2:-2]
                            pos = pos.split("), (")
                            for poly_str in pos:
                                points = poly_str.split(", ")
                                poly = shapely.geometry.Polygon([(points[0], points[1]),
                                                                (points[2], points[1]),
                                                                (points[2], points[3]),
                                                                (points[0], points[3])])
                                text_bounds.append(poly)
                    else:
                        text_bounds = None
                else:
                    raise ValueError('Invalid method for text-extraction')

                # update only makes sense if we have found text boxes
                if text is not None:

                    # merge both text position in one data dict
                    data = {
                        "text": text,
                        "text_bounds": text_bounds
                    }

                    # update text in table
                    success = utie.update_table_images_extracted(image_id, "text", data,
                                                                 overwrite=overwrite,
                                                                 catch=catch, verbose=verbose, pbar=pbar)

                    # check if it was possible to update the table
                    if success is None:
                        fail_manager.update_failed_id(image_id, "update_table_images_extracted:text")
                        continue

        if debug_extract_params:
            # check if we already have extracted all other information
            if table_data is not None and \
                    table_data["focal_length"] is not None and \
                    table_data["lens_cone"] is not None and \
                    table_data["awar"] is not None and \
                    overwrite is False:
                p.print_v(f"Information is already extracted for {image_id}",
                          verbose=verbose, pbar=pbar)
            else:

                # get text if undefined
                if text is None:
                    sql_string = f"SELECT text_content FROM images_extracted WHERE " \
                                 f"image_id='{image_id}'"
                    table_data = ctd.get_data_from_db(sql_string, catch=catch, verbose=verbose, pbar=pbar)
                    text = table_data.iloc[0]['text_content']

                # try to extract some information about the camera from the images based on the text
                if text is not None:
                    data = eip.extract_image_parameters(text, image_id,
                                                        catch=catch, verbose=verbose, pbar=pbar)
                else:
                    data = None

                if data is None:
                    fail_manager.update_failed_id(image_id, "extract_image_parameters")
                    continue

                success = utie.update_table_images_extracted(image_id, "params", data,
                                                             overwrite=overwrite,
                                                             catch=catch, verbose=verbose, pbar=pbar)

                # check if it was possible to update the table
                if success is None:
                    fail_manager.update_failed_id(image_id, "update_table_images_extracted:params")
                    continue

        if debug_extract_complexity:

            if overwrite is False and table_data['complexity'] is not None:
                p.print_v("Complexity already calculated", verbose=verbose, pbar=pbar)
                continue

            # check if we need to load the image
            if image is None:

                # load the image
                image = liff.load_image_from_file(image_id, catch=catch, verbose=verbose, pbar=pbar)

            image_no_borders = rb.remove_borders(image, image_id,
                                                 catch=catch, verbose=verbose, pbar=pbar)

            # check if we could load the image
            if image_no_borders is None:
                fail_manager.update_failed_id(image_id, "load_image_from_file")
                continue

            complexity = cic.calc_image_complexity(image_no_borders, highscore=complexity_highscore,
                                                   catch=catch, verbose=verbose, pbar=pbar)

            if complexity is None:
                fail_manager.update_failed_id(image_id, "calc_complexity")

            success = utie.update_table_images_extracted(image_id, "complexity", complexity,
                                                         overwrite=overwrite,
                                                         catch=catch, verbose=verbose, pbar=pbar)

            # check if it was possible to update the table
            if success is None:
                fail_manager.update_failed_id(image_id, "update_table_images_extracted:complexity")
                continue

        # if everything worked remove from failed ids
        fail_manager.remove_failed_id(image_id)

    # save the failed ids
    fail_manager.save_csv()

    p.print_v(f"Finished: extract_for_all_images ({len(image_ids)} images)", verbose=verbose)


if __name__ == "__main__":

    # img_ids = ['CA154931L0077', 'CA154931L0078', 'CA171831L0059', 'CA171831L0060', 'CA171831L0061', 'CA171831L0062', 'CA171831L0063', 'CA171831L0064', 'CA171831L0065', 'CA171831L0066', 'CA171831L0067', 'CA171831L0068', 'CA171831L0069', 'CA171831L0070', 'CA171831L0071', 'CA171831L0072', 'CA171831L0073', 'CA171831L0074', 'CA171831L0075', 'CA171831L0076', 'CA171831L0077', 'CA171831L0078', 'CA171831L0079', 'CA171831L0080', 'CA171831L0081', 'CA171831L0082', 'CA171831L0083', 'CA171831L0084', 'CA171831L0085', 'CA171831L0086', 'CA171831L0087', 'CA171831L0088', 'CA171931L0089', 'CA171931L0090', 'CA171931L0091', 'CA171931L0092', 'CA171931L0093', 'CA171931L0094', 'CA171931L0095', 'CA171931L0096', 'CA171931L0097', 'CA171931L0098', 'CA171931L0099', 'CA171931L0100', 'CA171931L0101', 'CA171931L0102', 'CA171931L0103', 'CA171931L0104', 'CA171931L0105', 'CA171931L0106', 'CA171931L0107', 'CA171931L0108', 'CA171931L0109', 'CA171931L0110', 'CA171931L0111', 'CA171931L0112', 'CA171931L0113', 'CA171931L0114', 'CA171931L0115', 'CA171931L0116', 'CA172031L0232', 'CA172031L0233', 'CA172031L0234', 'CA172031L0235', 'CA172031L0236', 'CA172031L0237', 'CA172031L0238', 'CA172031L0239', 'CA172031L0240', 'CA172031L0241', 'CA172031L0242', 'CA172031L0243', 'CA172031L0244', 'CA172031L0245', 'CA172031L0246', 'CA172031L0247', 'CA172031L0248', 'CA172031L0249', 'CA172031L0250', 'CA172031L0251', 'CA172031L0252', 'CA172031L0253', 'CA172031L0254', 'CA172031L0255', 'CA172031L0256', 'CA172031L0257', 'CA172031L0258', 'CA172031L0259', 'CA172031L0260', 'CA172031L0261', 'CA172031L0262', 'CA172031L0263', 'CA172031L0264', 'CA172031L0265', 'CA172031L0266', 'CA172031L0267', 'CA172031L0268', 'CA172031L0269', 'CA172031L0270', 'CA172031L0271', 'CA172031L0272', 'CA172031L0273', 'CA172031L0274', 'CA172031L0275', 'CA172131L0276', 'CA172131L0277', 'CA172131L0278', 'CA172131L0279', 'CA172131L0280', 'CA172131L0281', 'CA172131L0282', 'CA172131L0283', 'CA172131L0284', 'CA172131L0285', 'CA172131L0286', 'CA172131L0287', 'CA172131L0288', 'CA172131L0289', 'CA172231L0066', 'CA172231L0067', 'CA172231L0068', 'CA172231L0069', 'CA172231L0070', 'CA172231L0071', 'CA172231L0072', 'CA172231L0073', 'CA172231L0074', 'CA172231L0075', 'CA172231L0076', 'CA172231L0077', 'CA172231L0078', 'CA172231L0079', 'CA172231L0080', 'CA172231L0081', 'CA172231L0082', 'CA172231L0083', 'CA172231L0084', 'CA172231L0085', 'CA172231L0086', 'CA172231L0087', 'CA172231L0088', 'CA172231L0089', 'CA172231L0090', 'CA172231L0091', 'CA172231L0092', 'CA172331L0153', 'CA172331L0154', 'CA172331L0155', 'CA172331L0156', 'CA172331L0157', 'CA172331L0158', 'CA172331L0159', 'CA172331L0160', 'CA172331L0161', 'CA172331L0162', 'CA172331L0163', 'CA172331L0164', 'CA172331L0165', 'CA172331L0166', 'CA172331L0167', 'CA172331L0168', 'CA172331L0169', 'CA172331L0170', 'CA172331L0171', 'CA172331L0172', 'CA172331L0173', 'CA172331L0174', 'CA172331L0175', 'CA172331L0176', 'CA172331L0177', 'CA172331L0178', 'CA172331L0179', 'CA172431L0252', 'CA172431L0253', 'CA172431L0254', 'CA172431L0255', 'CA172431L0256', 'CA172431L0257', 'CA172431L0258', 'CA172431L0259', 'CA172431L0260', 'CA172431L0261', 'CA172431L0262', 'CA172431L0263', 'CA172431L0264', 'CA172431L0265', 'CA172431L0266', 'CA172431L0267', 'CA172431L0268', 'CA172431L0269', 'CA172431L0270', 'CA172431L0271', 'CA172431L0272', 'CA172431L0273', 'CA172431L0274', 'CA172531L0001', 'CA172531L0002', 'CA172531L0003', 'CA172531L0004', 'CA172531L0005', 'CA172531L0006', 'CA172531L0007', 'CA172531L0008', 'CA172531L0009', 'CA172531L0010', 'CA172531L0011', 'CA172531L0012', 'CA172531L0013', 'CA172531L0014', 'CA172531L0015', 'CA172531L0016', 'CA172531L0017', 'CA172531L0018', 'CA189331L0280', 'CA189331L0281', 'CA189331L0282', 'CA189331L0283', 'CA189331L0284', 'CA189331L0285', 'CA189331L0286', 'CA189331L0287', 'CA189331L0288', 'CA189331L0289', 'CA189331L0290', 'CA189331L0291', 'CA189331L0292', 'CA189331L0293', 'CA189331L0294', 'CA189331L0295', 'CA189331L0296', 'CA189331L0297', 'CA189331L0298', 'CA189331L0299', 'CA189331L0300', 'CA189331L0301', 'CA189331L0302', 'CA189331L0303', 'CA189331L0304', 'CA189331L0305', 'CA189331L0306', 'CA189331L0307', 'CA189331L0308', 'CA189331L0309', 'CA189331L0310', 'CA189331L0311', 'CA189331L0312', 'CA189331L0313', 'CA189331L0314', 'CA189331L0315', 'CA189331L0316', 'CA189331L0317', 'CA189831L0239', 'CA189831L0240', 'CA189831L0241', 'CA189831L0242', 'CA189831L0243', 'CA189831L0244', 'CA189831L0245', 'CA189831L0246', 'CA189831L0247', 'CA189831L0248', 'CA189831L0249', 'CA189831L0250', 'CA189831L0251', 'CA189831L0252', 'CA189831L0253', 'CA189831L0254', 'CA189831L0255', 'CA189831L0256', 'CA189831L0257', 'CA189831L0258', 'CA189831L0259', 'CA189831L0260', 'CA189831L0261', 'CA189831L0262', 'CA189831L0263', 'CA189831L0264', 'CA189831L0265', 'CA189831L0266', 'CA189831L0267', 'CA189831L0268', 'CA189831L0269', 'CA189831L0270', 'CA189831L0271', 'CA189831L0272', 'CA189831L0273', 'CA189831L0274', 'CA189831L0275', 'CA189831L0276', 'CA189831L0277', 'CA189831L0278', 'CA189831L0279', 'CA189831L0280', 'CA189831L0281', 'CA189831L0282', 'CA198231L0001', 'CA198231L0002', 'CA198231L0003', 'CA198231L0004', 'CA198231L0005', 'CA198231L0006', 'CA198231L0007', 'CA198231L0008', 'CA198231L0009', 'CA198231L0010', 'CA198231L0011', 'CA198231L0012', 'CA198231L0013', 'CA198231L0014', 'CA198231L0015', 'CA198231L0016', 'CA198231L0017', 'CA198231L0018', 'CA198231L0019', 'CA198231L0020', 'CA198231L0021', 'CA198231L0022', 'CA198231L0023', 'CA198231L0024', 'CA198231L0025', 'CA198231L0026', 'CA198231L0027', 'CA198231L0028', 'CA198231L0029', 'CA198231L0030', 'CA198231L0031', 'CA198231L0032', 'CA198231L0033', 'CA198231L0034', 'CA198231L0035', 'CA198231L0036', 'CA198231L0037', 'CA198231L0038', 'CA203931L0181', 'CA203931L0182', 'CA203931L0183', 'CA203931L0184', 'CA203931L0185', 'CA203931L0186', 'CA203931L0187', 'CA203931L0188', 'CA203931L0189', 'CA203931L0190', 'CA203931L0191', 'CA203931L0192', 'CA203931L0193', 'CA203931L0194', 'CA203931L0195', 'CA203931L0196', 'CA203931L0197', 'CA203931L0198', 'CA203931L0199', 'CA203931L0200', 'CA203931L0201', 'CA203931L0202', 'CA203931L0203', 'CA203931L0204', 'CA203931L0205', 'CA203931L0206', 'CA203931L0207', 'CA203931L0208', 'CA203931L0209', 'CA203931L0210', 'CA203931L0211', 'CA203931L0212', 'CA203931L0213', 'CA203931L0214', 'CA203931L0215', 'CA203931L0216', 'CA203931L0217', 'CA203931L0218', 'CA203931L0219', 'CA203931L0220', 'CA203931L0221', 'CA203931L0222', 'CA203931L0223', 'CA203931L0224', 'CA203931L0225', 'CA203931L0226', 'CA203931L0227', 'CA203931L0228', 'CA203931L0229', 'CA203931L0230', 'CA203931L0231', 'CA203931L0232', 'CA203931L0233', 'CA203931L0234', 'CA203931L0235', 'CA203931L0236', 'CA203931L0237', 'CA203931L0238', 'CA203931L0239', 'CA203931L0240', 'CA203931L0241', 'CA203931L0242', 'CA203931L0243', 'CA203931L0244', 'CA203931L0245', 'CA203931L0246', 'CA203931L0247', 'CA203931L0248', 'CA203931L0249', 'CA203931L0250', 'CA203931L0251', 'CA203931L0252', 'CA203931L0253', 'CA203931L0254', 'CA203931L0255', 'CA203931L0256', 'CA203931L0257', 'CA203931L0258', 'CA203931L0259', 'CA203931L0260', 'CA203931L0261', 'CA203931L0262', 'CA203931L0263', 'CA203931L0264', 'CA203931L0265', 'CA203931L0266', 'CA203931L0267', 'CA203931L0268', 'CA203931L0269', 'CA203931L0270', 'CA203931L0271', 'CA203931L0272', 'CA203931L0273', 'CA203931L0274', 'CA203931L0275', 'CA203931L0276', 'CA203931L0277', 'CA203931L0278', 'CA203931L0279', 'CA203931L0280', 'CA203931L0281', 'CA203931L0282', 'CA203931L0283', 'CA203931L0284', 'CA203931L0285', 'CA203931L0286', 'CA203931L0287', 'CA203931L0288', 'CA203931L0289', 'CA203931L0290', 'CA203931L0291', 'CA203931L0292', 'CA203931L0293', 'CA203931L0294', 'CA203931L0295', 'CA203931L0296', 'CA203931L0297', 'CA203931L0298', 'CA203931L0299', 'CA203931L0300', 'CA203931L0301', 'CA203931L0302', 'CA203931L0303', 'CA203931L0304', 'CA203931L0305', 'CA203931L0306', 'CA203931L0307', 'CA204031L0308', 'CA204031L0309', 'CA204031L0310', 'CA204031L0311', 'CA204031L0312', 'CA204031L0313', 'CA204031L0314', 'CA204031L0315', 'CA204031L0316', 'CA204031L0317', 'CA204031L0318', 'CA204031L0319', 'CA204031L0320', 'CA204031L0321', 'CA204031L0322', 'CA204031L0323', 'CA204031L0324', 'CA204031L0325', 'CA204031L0326', 'CA204031L0327', 'CA204031L0328', 'CA204031L0329', 'CA204031L0330', 'CA204031L0331', 'CA204031L0332', 'CA204031L0333', 'CA204031L0334', 'CA204031L0335', 'CA204031L0336', 'CA204031L0337', 'CA204031L0338', 'CA204031L0339', 'CA204031L0340', 'CA204031L0341', 'CA204031L0342', 'CA204031L0343', 'CA204031L0344', 'CA204031L0345', 'CA204031L0346', 'CA204031L0347', 'CA204031L0348', 'CA204031L0349', 'CA204031L0350', 'CA204031L0351', 'CA204031L0352', 'CA204031L0353', 'CA204031L0354', 'CA204031L0355', 'CA204031L0356', 'CA204031L0357', 'CA204031L0358', 'CA204031L0359', 'CA204031L0360', 'CA204031L0361', 'CA204031L0362', 'CA204031L0363', 'CA204031L0364', 'CA204031L0365', 'CA204031L0366', 'CA204031L0367', 'CA204031L0368', 'CA204031L0369', 'CA204031L0370', 'CA204031L0371', 'CA204031L0372', 'CA204031L0373', 'CA204031L0374', 'CA204031L0375', 'CA204031L0376', 'CA204031L0377', 'CA204031L0378', 'CA204031L0379', 'CA204031L0380', 'CA204031L0381', 'CA204031L0382', 'CA204031L0383', 'CA204031L0384', 'CA204031L0385', 'CA204031L0386', 'CA204031L0387', 'CA204031L0388', 'CA204031L0389', 'CA204031L0390', 'CA204031L0391', 'CA204031L0392', 'CA204031L0393', 'CA204031L0394', 'CA204031L0395', 'CA204031L0396', 'CA204031L0397', 'CA204031L0398', 'CA204031L0399', 'CA204031L0400', 'CA204031L0401', 'CA204031L0402', 'CA204031L0403', 'CA204031L0404', 'CA204031L0405', 'CA204031L0406', 'CA204031L0407', 'CA204031L0408', 'CA204031L0409', 'CA204031L0410', 'CA204031L0411', 'CA204031L0412', 'CA204031L0413', 'CA204031L0414', 'CA204031L0415', 'CA204031L0416', 'CA204031L0417', 'CA204031L0418', 'CA204031L0419', 'CA204031L0420', 'CA204131L0035', 'CA204131L0036', 'CA204131L0037', 'CA204131L0038', 'CA204131L0039', 'CA204131L0040', 'CA204131L0041', 'CA204131L0042', 'CA204131L0043', 'CA204131L0044', 'CA204131L0045', 'CA204131L0046', 'CA204131L0047', 'CA204131L0048', 'CA204131L0049', 'CA204131L0050', 'CA204131L0051', 'CA204131L0052', 'CA204131L0053', 'CA204131L0054', 'CA204131L0055', 'CA204131L0056', 'CA204131L0057', 'CA204131L0058', 'CA204131L0059', 'CA204131L0060', 'CA204131L0061', 'CA204131L0062', 'CA204131L0063', 'CA204131L0064', 'CA204131L0065', 'CA204131L0066', 'CA204131L0067', 'CA204131L0068', 'CA204131L0069', 'CA204131L0070', 'CA204131L0071', 'CA204131L0072', 'CA204131L0073', 'CA204131L0074', 'CA204131L0075', 'CA204131L0076', 'CA204131L0077', 'CA204131L0078', 'CA204131L0079', 'CA204131L0080', 'CA204131L0081', 'CA204131L0082', 'CA204131L0083', 'CA204131L0084', 'CA204131L0085', 'CA204131L0086', 'CA204131L0087', 'CA204131L0088', 'CA204131L0089', 'CA204131L0090', 'CA204131L0091', 'CA204131L0092', 'CA204131L0093', 'CA204131L0094', 'CA204131L0095', 'CA204131L0096', 'CA204131L0097', 'CA204131L0098', 'CA204131L0099', 'CA204131L0100', 'CA204131L0101', 'CA204131L0102', 'CA204131L0103', 'CA204131L0104', 'CA204131L0105', 'CA204131L0106', 'CA204131L0107', 'CA204131L0108', 'CA204131L0109', 'CA204131L0110', 'CA204131L0111', 'CA204131L0112', 'CA204131L0113', 'CA204131L0114', 'CA204131L0115', 'CA204231L0146', 'CA204231L0147', 'CA204231L0148', 'CA204231L0149', 'CA204231L0150', 'CA204231L0151', 'CA204231L0152', 'CA204231L0153', 'CA204231L0154', 'CA204231L0155', 'CA204231L0156', 'CA204231L0157', 'CA204231L0158', 'CA204231L0159', 'CA204231L0160', 'CA204231L0161', 'CA204231L0162', 'CA204231L0163', 'CA204231L0164', 'CA204231L0165', 'CA204231L0166', 'CA204231L0167', 'CA204231L0168', 'CA204231L0169', 'CA204231L0170', 'CA204231L0171', 'CA204231L0172', 'CA204231L0173', 'CA204231L0174', 'CA204231L0175', 'CA204331L0238', 'CA204331L0239', 'CA204331L0240', 'CA204331L0241', 'CA204331L0242', 'CA204331L0243', 'CA204331L0244', 'CA204331L0245', 'CA204331L0246', 'CA204331L0247', 'CA204331L0248', 'CA204331L0249', 'CA204331L0250', 'CA204331L0251', 'CA204331L0252', 'CA204331L0253', 'CA204331L0254', 'CA512031L0003', 'CA512031L0004', 'CA512031L0005', 'CA512031L0006', 'CA512031L0007', 'CA512031L0008', 'CA512031L0009', 'CA512031L0010', 'CA512031L0011', 'CA512031L0012', 'CA512031L0013', 'CA512031L0014', 'CA512031L0015', 'CA512031L0016', 'CA512031L0017', 'CA512031L0018', 'CA512031L0019', 'CA512031L0020', 'CA512031L0021', 'CA512031L0022', 'CA512031L0023', 'CA512031L0024', 'CA512031L0025', 'CA512031L0026', 'CA512031L0027', 'CA512031L0028', 'CA512231L0001', 'CA512231L0002', 'CA512231L0003', 'CA512231L0004', 'CA512231L0005', 'CA512231L0006', 'CA512231L0007', 'CA512231L0008', 'CA512231L0009', 'CA512231L0010', 'CA512231L0011', 'CA512231L0012', 'CA512231L0013', 'CA512231L0014', 'CA512231L0015', 'CA512331L0001', 'CA512331L0002', 'CA512331L0003', 'CA512331L0004', 'CA512331L0005', 'CA512331L0006', 'CA512331L0007', 'CA512331L0008', 'CA512331L0009', 'CA512331L0010', 'CA512331L0011', 'CA512331L0012', 'CA512331L0013', 'CA512331L0014', 'CA512331L0015', 'CA512331L0016', 'CA512331L0017', 'CA512331L0018', 'CA512331L0019', 'CA512331L0020', 'CA512331L0021', 'CA512331L0022', 'CA512331L0023', 'CA512331L0024', 'CA512331L0025', 'CA512331L0026', 'CA512331L0027', 'CA512331L0028', 'CA512331L0029', 'CA512331L0030', 'CA512331L0031', 'CA512331L0032', 'CA512331L0033', 'CA512331L0034', 'CA512331L0035', 'CA512331L0036', 'CA512331L0037', 'CA512331L0038', 'CA512331L0039', 'CA512331L0040', 'CA512331L0041', 'CA512331L0042', 'CA512331L0043', 'CA512331L0044', 'CA512331L0045', 'CA512331L0046', 'CA512331L0047', 'CA512331L0048', 'CA512331L0049', 'CA512331L0050', 'CA512331L0051', 'CA512331L0052', 'CA512331L0053', 'CA512331L0054', 'CA512331L0055', 'CA512331L0056', 'CA512331L0057', 'CA512331L0058', 'CA512331L0059', 'CA512331L0060', 'CA512331L0061', 'CA512331L0062', 'CA512331L0063', 'CA512331L0064', 'CA512331L0065', 'CA512331L0066', 'CA512331L0067', 'CA512331L0068', 'CA512331L0069', 'CA512331L0070', 'CA512331L0071', 'CA512331L0072', 'CA512331L0073', 'CA512331L0074', 'CA512331L0075', 'CA512331L0076', 'CA512331L0077', 'CA512331L0078', 'CA512331L0079', 'CA512331L0080', 'CA512331L0081', 'CA512331L0082', 'CA512331L0083', 'CA512331L0084', 'CA512331L0085', 'CA512331L0086', 'CA512331L0087', 'CA512331L0088', 'CA512331L0089', 'CA512331L0090', 'CA512331L0091', 'CA512331L0092', 'CA512331L0093', 'CA512331L0094', 'CA512331L0095', 'CA512331L0096', 'CA512331L0097', 'CA512331L0098', 'CA512331L0099', 'CA512331L0100', 'CA512331L0101', 'CA512331L0102', 'CA512331L0103', 'CA512331L0104', 'CA512331L0105', 'CA512331L0106', 'CA512331L0107', 'CA512331L0108', 'CA512331L0109', 'CA512331L0110', 'CA512331L0111', 'CA512331L0112', 'CA512331L0113', 'CA512331L0114', 'CA512331L0115', 'CA512331L0116', 'CA512331L0117', 'CA512331L0118', 'CA512331L0119', 'CA512331L0120', 'CA512331L0121', 'CA512331L0122', 'CA512331L0123', 'CA512331L0124', 'CA512331L0125', 'CA512331L0126', 'CA512331L0127', 'CA512331L0128', 'CA512331L0129', 'CA512331L0130', 'CA512331L0131', 'CA512331L0132', 'CA512331L0133', 'CA512331L0134', 'CA512331L0135', 'CA512331L0136', 'CA512331L0137', 'CA512331L0138', 'CA512331L0139', 'CA512331L0140', 'CA512331L0141', 'CA512331L0142', 'CA512331L0143', 'CA512431L0005', 'CA512431L0006', 'CA512431L0007', 'CA512431L0008', 'CA512431L0009', 'CA512431L0010', 'CA512431L0011', 'CA512431L0012', 'CA512431L0013', 'CA512431L0014', 'CA512431L0015', 'CA512431L0016', 'CA512431L0017', 'CA512431L0018', 'CA512431L0019', 'CA512431L0020', 'CA512431L0021', 'CA512431L0022', 'CA512431L0023', 'CA512431L0024', 'CA512431L0025', 'CA512431L0026', 'CA512431L0027', 'CA512431L0028', 'CA512431L0029', 'CA512431L0030', 'CA512431L0031', 'CA512431L0032', 'CA512431L0033', 'CA512431L0034', 'CA512431L0035', 'CA512431L0036', 'CA512431L0037', 'CA512431L0038', 'CA512431L0039', 'CA512431L0040', 'CA512431L0041', 'CA512431L0042', 'CA512431L0043', 'CA512531L0044', 'CA512531L0045', 'CA512531L0046', 'CA512531L0047', 'CA512531L0048', 'CA512531L0049', 'CA512531L0050', 'CA512531L0051', 'CA512531L0052', 'CA512531L0053', 'CA512531L0054', 'CA512531L0055', 'CA512531L0056', 'CA512531L0057', 'CA512531L0058', 'CA512631L0095', 'CA512631L0094', 'CA512631L0093', 'CA512631L0092', 'CA512631L0091', 'CA512631L0090', 'CA512631L0089', 'CA512631L0088', 'CA512631L0087', 'CA512631L0086', 'CA512631L0085', 'CA512631L0084', 'CA512631L0083', 'CA512631L0082', 'CA512631L0081', 'CA512631L0080', 'CA512631L0079', 'CA512631L0078', 'CA512631L0077', 'CA512631L0076', 'CA512731L0096', 'CA512731L0097', 'CA512731L0098', 'CA512731L0099', 'CA512731L0100', 'CA512731L0101', 'CA512731L0102', 'CA512731L0103', 'CA512731L0104', 'CA512731L0105', 'CA512731L0106', 'CA512731L0107', 'CA512731L0108', 'CA512731L0109', 'CA512731L0110', 'CA512731L0111', 'CA512731L0112', 'CA512731L0113', 'CA512731L0114', 'CA512731L0115', 'CA512731L0116', 'CA512731L0117', 'CA512731L0118', 'CA512731L0119', 'CA512731L0120', 'CA512731L0121', 'CA512731L0122', 'CA512731L0123', 'CA512731L0124', 'CA512731L0125', 'CA512731L0126', 'CA512731L0127', 'CA512731L0128', 'CA512731L0129', 'CA512731L0130', 'CA512731L0131', 'CA512731L0132', 'CA512731L0133', 'CA512731L0134', 'CA512731L0135', 'CA512731L0136', 'CA154932V0077', 'CA154932V0078', 'CA171832V0059', 'CA171832V0060', 'CA171832V0061', 'CA171832V0062', 'CA171832V0063', 'CA171832V0064', 'CA171832V0065', 'CA171832V0066', 'CA171832V0067', 'CA171832V0068', 'CA171832V0069', 'CA171832V0070', 'CA171832V0071', 'CA171832V0072', 'CA171832V0073', 'CA171832V0074', 'CA171832V0075', 'CA171832V0076', 'CA171832V0077', 'CA171832V0078', 'CA171832V0079', 'CA171832V0080', 'CA171832V0081', 'CA171832V0082', 'CA171832V0083', 'CA171832V0084', 'CA171832V0085', 'CA171832V0086', 'CA171832V0087', 'CA171832V0088', 'CA171932V0089', 'CA171932V0090', 'CA171932V0091', 'CA171932V0092', 'CA171932V0093', 'CA171932V0094', 'CA171932V0095', 'CA171932V0096', 'CA171932V0097', 'CA171932V0098', 'CA171932V0099', 'CA171932V0100', 'CA171932V0101', 'CA171932V0102', 'CA171932V0103', 'CA171932V0104', 'CA171932V0105', 'CA171932V0106', 'CA171932V0107', 'CA171932V0108', 'CA171932V0109', 'CA171932V0110', 'CA171932V0111', 'CA171932V0112', 'CA171932V0113', 'CA171932V0114', 'CA171932V0115', 'CA171932V0116', 'CA172032V0232', 'CA172032V0233', 'CA172032V0234', 'CA172032V0235', 'CA172032V0236', 'CA172032V0237', 'CA172032V0238', 'CA172032V0239', 'CA172032V0240', 'CA172032V0241', 'CA172032V0242', 'CA172032V0243', 'CA172032V0244', 'CA172032V0245', 'CA172032V0246', 'CA172032V0247', 'CA172032V0248', 'CA172032V0249', 'CA172032V0250', 'CA172032V0251', 'CA172032V0252', 'CA172032V0253', 'CA172032V0254', 'CA172032V0255', 'CA172032V0256', 'CA172032V0257', 'CA172032V0258', 'CA172032V0259', 'CA172032V0260', 'CA172032V0261', 'CA172032V0262', 'CA172032V0263', 'CA172032V0264', 'CA172032V0265', 'CA172032V0266', 'CA172032V0267', 'CA172032V0268', 'CA172032V0269', 'CA172032V0270', 'CA172032V0271', 'CA172032V0272', 'CA172032V0273', 'CA172032V0274', 'CA172032V0275', 'CA172132V0276', 'CA172132V0277', 'CA172132V0278', 'CA172132V0279', 'CA172132V0280', 'CA172132V0281', 'CA172132V0282', 'CA172132V0283', 'CA172132V0284', 'CA172132V0285', 'CA172132V0286', 'CA172132V0287', 'CA172132V0288', 'CA172132V0289', 'CA172232V0066', 'CA172232V0067', 'CA172232V0068', 'CA172232V0069', 'CA172232V0070', 'CA172232V0071', 'CA172232V0072', 'CA172232V0073', 'CA172232V0074', 'CA172232V0075', 'CA172232V0076', 'CA172232V0077', 'CA172232V0078', 'CA172232V0079', 'CA172232V0080', 'CA172232V0081', 'CA172232V0082', 'CA172232V0083', 'CA172232V0084', 'CA172232V0085', 'CA172232V0086', 'CA172232V0087', 'CA172232V0088', 'CA172232V0089', 'CA172232V0090', 'CA172232V0091', 'CA172232V0092', 'CA172332V0153', 'CA172332V0154', 'CA172332V0155', 'CA172332V0156', 'CA172332V0157', 'CA172332V0158', 'CA172332V0159', 'CA172332V0160', 'CA172332V0161', 'CA172332V0162', 'CA172332V0163', 'CA172332V0164', 'CA172332V0165', 'CA172332V0166', 'CA172332V0167', 'CA172332V0168', 'CA172332V0169', 'CA172332V0170', 'CA172332V0171', 'CA172332V0172', 'CA172332V0173', 'CA172332V0174', 'CA172332V0175', 'CA172332V0176', 'CA172332V0177', 'CA172332V0178', 'CA172332V0179', 'CA172432V0252', 'CA172432V0253', 'CA172432V0254', 'CA172432V0255', 'CA172432V0256', 'CA172432V0257', 'CA172432V0258', 'CA172432V0259', 'CA172432V0260', 'CA172432V0261', 'CA172432V0262', 'CA172432V0263', 'CA172432V0264', 'CA172432V0265', 'CA172432V0266', 'CA172432V0267', 'CA172432V0268', 'CA172432V0269', 'CA172432V0270', 'CA172432V0271', 'CA172432V0272', 'CA172432V0273', 'CA172432V0274', 'CA172532V0001', 'CA172532V0002', 'CA172532V0003', 'CA172532V0004', 'CA172532V0005', 'CA172532V0006', 'CA172532V0007', 'CA172532V0008', 'CA172532V0009', 'CA172532V0010', 'CA172532V0011', 'CA172532V0012', 'CA172532V0013', 'CA172532V0014', 'CA172532V0015', 'CA172532V0016', 'CA172532V0017', 'CA172532V0018', 'CA189332V0280', 'CA189332V0281', 'CA189332V0282', 'CA189332V0283', 'CA189332V0284', 'CA189332V0285', 'CA189332V0286', 'CA189332V0287', 'CA189332V0288', 'CA189332V0289', 'CA189332V0290', 'CA189332V0291', 'CA189332V0292', 'CA189332V0293', 'CA189332V0294', 'CA189332V0295', 'CA189332V0296', 'CA189332V0297', 'CA189332V0298', 'CA189332V0299', 'CA189332V0300', 'CA189332V0301', 'CA189332V0302', 'CA189332V0303', 'CA189332V0304', 'CA189332V0305', 'CA189332V0306', 'CA189332V0307', 'CA189332V0308', 'CA189332V0309', 'CA189332V0310', 'CA189332V0311', 'CA189332V0312', 'CA189332V0313', 'CA189332V0314', 'CA189332V0315', 'CA189332V0316', 'CA189332V0317', 'CA189832V0239', 'CA189832V0240', 'CA189832V0241', 'CA189832V0242', 'CA189832V0243', 'CA189832V0244', 'CA189832V0245', 'CA189832V0246', 'CA189832V0247', 'CA189832V0248', 'CA189832V0249', 'CA189832V0250', 'CA189832V0251', 'CA189832V0252', 'CA189832V0253', 'CA189832V0254', 'CA189832V0255', 'CA189832V0256', 'CA189832V0257', 'CA189832V0258', 'CA189832V0259', 'CA189832V0260', 'CA189832V0261', 'CA189832V0262', 'CA189832V0263', 'CA189832V0264', 'CA189832V0265', 'CA189832V0266', 'CA189832V0267', 'CA189832V0268', 'CA189832V0269', 'CA189832V0270', 'CA189832V0271', 'CA189832V0272', 'CA189832V0273', 'CA189832V0274', 'CA189832V0275', 'CA189832V0276', 'CA189832V0277', 'CA189832V0278', 'CA189832V0279', 'CA189832V0280', 'CA189832V0281', 'CA189832V0282', 'CA198232V0001', 'CA198232V0002', 'CA198232V0003', 'CA198232V0004', 'CA198232V0005', 'CA198232V0006', 'CA198232V0007', 'CA198232V0008', 'CA198232V0009', 'CA198232V0010', 'CA198232V0011', 'CA198232V0012', 'CA198232V0013', 'CA198232V0014', 'CA198232V0015', 'CA198232V0016', 'CA198232V0017', 'CA198232V0018', 'CA198232V0019', 'CA198232V0020', 'CA198232V0021', 'CA198232V0022', 'CA198232V0023', 'CA198232V0024', 'CA198232V0025', 'CA198232V0026', 'CA198232V0027', 'CA198232V0028', 'CA198232V0029', 'CA198232V0030', 'CA198232V0031', 'CA198232V0032', 'CA198232V0033', 'CA198232V0034', 'CA198232V0035', 'CA198232V0036', 'CA198232V0037', 'CA198232V0038', 'CA203932V0181', 'CA203932V0182', 'CA203932V0183', 'CA203932V0184', 'CA203932V0185', 'CA203932V0186', 'CA203932V0187', 'CA203932V0188', 'CA203932V0189', 'CA203932V0190', 'CA203932V0191', 'CA203932V0192', 'CA203932V0193', 'CA203932V0194', 'CA203932V0195', 'CA203932V0196', 'CA203932V0197', 'CA203932V0198', 'CA203932V0199', 'CA203932V0200', 'CA203932V0201', 'CA203932V0202', 'CA203932V0203', 'CA203932V0204', 'CA203932V0205', 'CA203932V0206', 'CA203932V0207', 'CA203932V0208', 'CA203932V0209', 'CA203932V0210', 'CA203932V0211', 'CA203932V0212', 'CA203932V0213', 'CA203932V0214', 'CA203932V0215', 'CA203932V0216', 'CA203932V0217', 'CA203932V0218', 'CA203932V0219', 'CA203932V0220', 'CA203932V0221', 'CA203932V0222', 'CA203932V0223', 'CA203932V0224', 'CA203932V0225', 'CA203932V0226', 'CA203932V0227', 'CA203932V0228', 'CA203932V0229', 'CA203932V0230', 'CA203932V0231', 'CA203932V0232', 'CA203932V0233', 'CA203932V0234', 'CA203932V0235', 'CA203932V0236', 'CA203932V0237', 'CA203932V0238', 'CA203932V0239', 'CA203932V0240', 'CA203932V0241', 'CA203932V0242', 'CA203932V0243', 'CA203932V0244', 'CA203932V0245', 'CA203932V0246', 'CA203932V0247', 'CA203932V0248', 'CA203932V0249', 'CA203932V0250', 'CA203932V0251', 'CA203932V0252', 'CA203932V0253', 'CA203932V0254', 'CA203932V0255', 'CA203932V0256', 'CA203932V0257', 'CA203932V0258', 'CA203932V0259', 'CA203932V0260', 'CA203932V0261', 'CA203932V0262', 'CA203932V0263', 'CA203932V0264', 'CA203932V0265', 'CA203932V0266', 'CA203932V0267', 'CA203932V0268', 'CA203932V0269', 'CA203932V0270', 'CA203932V0271', 'CA203932V0272', 'CA203932V0273', 'CA203932V0274', 'CA203932V0275', 'CA203932V0276', 'CA203932V0277', 'CA203932V0278', 'CA203932V0279', 'CA203932V0280', 'CA203932V0281', 'CA203932V0282', 'CA203932V0283', 'CA203932V0284', 'CA203932V0285', 'CA203932V0286', 'CA203932V0287', 'CA203932V0288', 'CA203932V0289', 'CA203932V0290', 'CA203932V0291', 'CA203932V0292', 'CA203932V0293', 'CA203932V0294', 'CA203932V0295', 'CA203932V0296', 'CA203932V0297', 'CA203932V0298', 'CA203932V0299', 'CA203932V0300', 'CA203932V0301', 'CA203932V0302', 'CA203932V0303', 'CA203932V0304', 'CA203932V0305', 'CA203932V0306', 'CA203932V0307', 'CA204032V0308', 'CA204032V0309', 'CA204032V0310', 'CA204032V0311', 'CA204032V0312', 'CA204032V0313', 'CA204032V0314', 'CA204032V0315', 'CA204032V0316', 'CA204032V0317', 'CA204032V0318', 'CA204032V0319', 'CA204032V0320', 'CA204032V0321', 'CA204032V0322', 'CA204032V0323', 'CA204032V0324', 'CA204032V0325', 'CA204032V0326', 'CA204032V0327', 'CA204032V0328', 'CA204032V0329', 'CA204032V0330', 'CA204032V0331', 'CA204032V0332', 'CA204032V0333', 'CA204032V0334', 'CA204032V0335', 'CA204032V0336', 'CA204032V0337', 'CA204032V0338', 'CA204032V0339', 'CA204032V0340', 'CA204032V0341', 'CA204032V0342', 'CA204032V0343', 'CA204032V0344', 'CA204032V0345', 'CA204032V0346', 'CA204032V0347', 'CA204032V0348', 'CA204032V0349', 'CA204032V0350', 'CA204032V0351', 'CA204032V0352', 'CA204032V0353', 'CA204032V0354', 'CA204032V0355', 'CA204032V0356', 'CA204032V0357', 'CA204032V0358', 'CA204032V0359', 'CA204032V0360', 'CA204032V0361', 'CA204032V0362', 'CA204032V0363', 'CA204032V0364', 'CA204032V0365', 'CA204032V0366', 'CA204032V0367', 'CA204032V0368', 'CA204032V0369', 'CA204032V0370', 'CA204032V0371', 'CA204032V0372', 'CA204032V0373', 'CA204032V0374', 'CA204032V0375', 'CA204032V0376', 'CA204032V0377', 'CA204032V0378', 'CA204032V0379', 'CA204032V0380', 'CA204032V0381', 'CA204032V0382', 'CA204032V0383', 'CA204032V0384', 'CA204032V0385', 'CA204032V0386', 'CA204032V0387', 'CA204032V0388', 'CA204032V0389', 'CA204032V0390', 'CA204032V0391', 'CA204032V0392', 'CA204032V0393', 'CA204032V0394', 'CA204032V0395', 'CA204032V0396', 'CA204032V0397', 'CA204032V0398', 'CA204032V0399', 'CA204032V0400', 'CA204032V0401', 'CA204032V0402', 'CA204032V0403', 'CA204032V0404', 'CA204032V0405', 'CA204032V0406', 'CA204032V0407', 'CA204032V0408', 'CA204032V0409', 'CA204032V0410', 'CA204032V0411', 'CA204032V0412', 'CA204032V0413', 'CA204032V0414', 'CA204032V0415', 'CA204032V0416', 'CA204032V0417', 'CA204032V0418', 'CA204032V0419', 'CA204032V0420', 'CA204132V0035', 'CA204132V0036', 'CA204132V0037', 'CA204132V0038', 'CA204132V0039', 'CA204132V0040', 'CA204132V0041', 'CA204132V0042', 'CA204132V0043', 'CA204132V0044', 'CA204132V0045', 'CA204132V0046', 'CA204132V0047', 'CA204132V0048', 'CA204132V0049', 'CA204132V0050', 'CA204132V0051', 'CA204132V0052', 'CA204132V0053', 'CA204132V0054', 'CA204132V0055', 'CA204132V0056', 'CA204132V0057', 'CA204132V0058', 'CA204132V0059', 'CA204132V0060', 'CA204132V0061', 'CA204132V0062', 'CA204132V0063', 'CA204132V0064', 'CA204132V0065', 'CA204132V0066', 'CA204132V0067', 'CA204132V0068', 'CA204132V0069', 'CA204132V0070', 'CA204132V0071', 'CA204132V0072', 'CA204132V0073', 'CA204132V0074', 'CA204132V0075', 'CA204132V0076', 'CA204132V0077', 'CA204132V0078', 'CA204132V0079', 'CA204132V0080', 'CA204132V0081', 'CA204132V0082', 'CA204132V0083', 'CA204132V0084', 'CA204132V0085', 'CA204132V0086', 'CA204132V0087', 'CA204132V0088', 'CA204132V0089', 'CA204132V0090', 'CA204132V0091', 'CA204132V0092', 'CA204132V0093', 'CA204132V0094', 'CA204132V0095', 'CA204132V0096', 'CA204132V0097', 'CA204132V0098', 'CA204132V0099', 'CA204132V0100', 'CA204132V0101', 'CA204132V0102', 'CA204132V0103', 'CA204132V0104', 'CA204132V0105', 'CA204132V0106', 'CA204132V0107', 'CA204132V0108', 'CA204132V0109', 'CA204132V0110', 'CA204132V0111', 'CA204132V0112', 'CA204132V0113', 'CA204132V0114', 'CA204132V0115', 'CA204232V0146', 'CA204232V0147', 'CA204232V0148', 'CA204232V0149', 'CA204232V0150', 'CA204232V0151', 'CA204232V0152', 'CA204232V0153', 'CA204232V0154', 'CA204232V0155', 'CA204232V0156', 'CA204232V0157', 'CA204232V0158', 'CA204232V0159', 'CA204232V0160', 'CA204232V0161', 'CA204232V0162', 'CA204232V0163', 'CA204232V0164', 'CA204232V0165', 'CA204232V0166', 'CA204232V0167', 'CA204232V0168', 'CA204232V0169', 'CA204232V0170', 'CA204232V0171', 'CA204232V0172', 'CA204232V0173', 'CA204232V0174', 'CA204232V0175', 'CA204332V0238', 'CA204332V0239', 'CA204332V0240', 'CA204332V0241', 'CA204332V0242', 'CA204332V0243', 'CA204332V0244', 'CA204332V0245', 'CA204332V0246', 'CA204332V0247', 'CA204332V0248', 'CA204332V0249', 'CA204332V0250', 'CA204332V0251', 'CA204332V0252', 'CA204332V0253', 'CA204332V0254', 'CA512032V0002', 'CA512032V0003', 'CA512032V0004', 'CA512032V0005', 'CA512032V0006', 'CA512032V0007', 'CA512032V0008', 'CA512032V0009', 'CA512032V0010', 'CA512032V0011', 'CA512032V0012', 'CA512032V0013', 'CA512032V0014', 'CA512032V0015', 'CA512032V0016', 'CA512032V0017', 'CA512032V0018', 'CA512032V0019', 'CA512032V0020', 'CA512032V0021', 'CA512032V0022', 'CA512032V0023', 'CA512032V0024', 'CA512032V0025', 'CA512032V0026', 'CA512032V0027', 'CA512032V0028', 'CA512232V0001', 'CA512232V0002', 'CA512232V0003', 'CA512232V0004', 'CA512232V0005', 'CA512232V0006', 'CA512232V0007', 'CA512232V0008', 'CA512232V0009', 'CA512232V0010', 'CA512232V0011', 'CA512232V0012', 'CA512232V0013', 'CA512232V0014', 'CA512232V0015', 'CA512332V0001', 'CA512332V0002', 'CA512332V0003', 'CA512332V0004', 'CA512332V0005', 'CA512332V0006', 'CA512332V0007', 'CA512332V0008', 'CA512332V0009', 'CA512332V0010', 'CA512332V0011', 'CA512332V0012', 'CA512332V0013', 'CA512332V0014', 'CA512332V0015', 'CA512332V0016', 'CA512332V0017', 'CA512332V0018', 'CA512332V0019', 'CA512332V0020', 'CA512332V0021', 'CA512332V0022', 'CA512332V0023', 'CA512332V0024', 'CA512332V0025', 'CA512332V0026', 'CA512332V0027', 'CA512332V0028', 'CA512332V0029', 'CA512332V0030', 'CA512332V0031', 'CA512332V0032', 'CA512332V0033', 'CA512332V0034', 'CA512332V0035', 'CA512332V0036', 'CA512332V0037', 'CA512332V0038', 'CA512332V0039', 'CA512332V0040', 'CA512332V0041', 'CA512332V0042', 'CA512332V0043', 'CA512332V0044', 'CA512332V0045', 'CA512332V0046', 'CA512332V0047', 'CA512332V0048', 'CA512332V0049', 'CA512332V0050', 'CA512332V0051', 'CA512332V0052', 'CA512332V0053', 'CA512332V0054', 'CA512332V0055', 'CA512332V0056', 'CA512332V0057', 'CA512332V0058', 'CA512332V0059', 'CA512332V0060', 'CA512332V0061', 'CA512332V0062', 'CA512332V0063', 'CA512332V0064', 'CA512332V0065', 'CA512332V0066', 'CA512332V0067', 'CA512332V0068', 'CA512332V0069', 'CA512332V0070', 'CA512332V0071', 'CA512332V0072', 'CA512332V0073', 'CA512332V0074', 'CA512332V0075', 'CA512332V0076', 'CA512332V0077', 'CA512332V0078', 'CA512332V0079', 'CA512332V0080', 'CA512332V0081', 'CA512332V0082', 'CA512332V0083', 'CA512332V0084', 'CA512332V0085', 'CA512332V0086', 'CA512332V0087', 'CA512332V0088', 'CA512332V0089', 'CA512332V0090', 'CA512332V0091', 'CA512332V0092', 'CA512332V0093', 'CA512332V0094', 'CA512332V0095', 'CA512332V0096', 'CA512332V0097', 'CA512332V0098', 'CA512332V0099', 'CA512332V0100', 'CA512332V0101', 'CA512332V0102', 'CA512332V0103', 'CA512332V0104', 'CA512332V0105', 'CA512332V0106', 'CA512332V0107', 'CA512332V0108', 'CA512332V0109', 'CA512332V0110', 'CA512332V0111', 'CA512332V0112', 'CA512332V0113', 'CA512332V0114', 'CA512332V0115', 'CA512332V0116', 'CA512332V0117', 'CA512332V0118', 'CA512332V0119', 'CA512332V0120', 'CA512332V0121', 'CA512332V0122', 'CA512332V0123', 'CA512332V0124', 'CA512332V0125', 'CA512332V0126', 'CA512332V0127', 'CA512332V0128', 'CA512332V0129', 'CA512332V0130', 'CA512332V0131', 'CA512332V0132', 'CA512332V0133', 'CA512332V0134', 'CA512332V0135', 'CA512332V0136', 'CA512332V0137', 'CA512332V0138', 'CA512332V0139', 'CA512332V0140', 'CA512332V0141', 'CA512332V0142', 'CA512332V0143', 'CA512432V0001', 'CA512432V0002', 'CA512432V0003', 'CA512432V0004', 'CA512432V0005', 'CA512432V0006', 'CA512432V0007', 'CA512432V0008', 'CA512432V0009', 'CA512432V0010', 'CA512432V0011', 'CA512432V0012', 'CA512432V0013', 'CA512432V0014', 'CA512432V0015', 'CA512432V0016', 'CA512432V0017', 'CA512432V0018', 'CA512432V0019', 'CA512432V0020', 'CA512432V0021', 'CA512432V0022', 'CA512432V0023', 'CA512432V0024', 'CA512432V0025', 'CA512432V0026', 'CA512432V0027', 'CA512432V0028', 'CA512432V0029', 'CA512432V0030', 'CA512432V0031', 'CA512432V0032', 'CA512432V0033', 'CA512432V0034', 'CA512432V0035', 'CA512432V0036', 'CA512432V0037', 'CA512432V0038', 'CA512432V0039', 'CA512432V0040', 'CA512432V0041', 'CA512432V0042', 'CA512432V0043', 'CA512532V0044', 'CA512532V0045', 'CA512532V0046', 'CA512532V0047', 'CA512532V0048', 'CA512532V0049', 'CA512532V0050', 'CA512532V0051', 'CA512532V0052', 'CA512532V0053', 'CA512532V0054', 'CA512532V0055', 'CA512532V0056', 'CA512532V0057', 'CA512532V0058', 'CA512632V0095', 'CA512632V0094', 'CA512632V0093', 'CA512632V0092', 'CA512632V0091', 'CA512632V0090', 'CA512632V0089', 'CA512632V0088', 'CA512632V0087', 'CA512632V0086', 'CA512632V0085', 'CA512632V0084', 'CA512632V0083', 'CA512632V0082', 'CA512632V0081', 'CA512632V0080', 'CA512632V0079', 'CA512632V0078', 'CA512632V0077', 'CA512632V0076', 'CA512732V0096', 'CA512732V0097', 'CA512732V0098', 'CA512732V0099', 'CA512732V0100', 'CA512732V0101', 'CA512732V0102', 'CA512732V0103', 'CA512732V0104', 'CA512732V0105', 'CA512732V0106', 'CA512732V0107', 'CA512732V0108', 'CA512732V0109', 'CA512732V0110', 'CA512732V0111', 'CA512732V0112', 'CA512732V0113', 'CA512732V0114', 'CA512732V0115', 'CA512732V0116', 'CA512732V0117', 'CA512732V0118', 'CA512732V0119', 'CA512732V0120', 'CA512732V0121', 'CA512732V0122', 'CA512732V0123', 'CA512732V0124', 'CA512732V0125', 'CA512732V0126', 'CA512732V0127', 'CA512732V0128', 'CA512732V0129', 'CA512732V0130', 'CA512732V0131', 'CA512732V0132', 'CA512732V0133', 'CA512732V0134', 'CA512732V0135', 'CA512732V0136', 'CA154933R0077', 'CA154933R0078', 'CA171833R0059', 'CA171833R0060', 'CA171833R0061', 'CA171833R0062', 'CA171833R0063', 'CA171833R0064', 'CA171833R0065', 'CA171833R0066', 'CA171833R0067', 'CA171833R0068', 'CA171833R0069', 'CA171833R0070', 'CA171833R0071', 'CA171833R0072', 'CA171833R0073', 'CA171833R0074', 'CA171833R0075', 'CA171833R0076', 'CA171833R0077', 'CA171833R0078', 'CA171833R0079', 'CA171833R0080', 'CA171833R0081', 'CA171833R0082', 'CA171833R0083', 'CA171833R0084', 'CA171833R0085', 'CA171833R0086', 'CA171833R0087', 'CA171833R0088', 'CA171933R0089', 'CA171933R0090', 'CA171933R0091', 'CA171933R0092', 'CA171933R0093', 'CA171933R0094', 'CA171933R0095', 'CA171933R0096', 'CA171933R0097', 'CA171933R0098', 'CA171933R0099', 'CA171933R0100', 'CA171933R0101', 'CA171933R0102', 'CA171933R0103', 'CA171933R0104', 'CA171933R0105', 'CA171933R0106', 'CA171933R0107', 'CA171933R0108', 'CA171933R0109', 'CA171933R0110', 'CA171933R0111', 'CA171933R0112', 'CA171933R0113', 'CA171933R0114', 'CA171933R0115', 'CA171933R0116', 'CA172033R0232', 'CA172033R0233', 'CA172033R0234', 'CA172033R0235', 'CA172033R0236', 'CA172033R0237', 'CA172033R0238', 'CA172033R0239', 'CA172033R0240', 'CA172033R0241', 'CA172033R0242', 'CA172033R0243', 'CA172033R0244', 'CA172033R0245', 'CA172033R0246', 'CA172033R0247', 'CA172033R0248', 'CA172033R0249', 'CA172033R0250', 'CA172033R0251', 'CA172033R0252', 'CA172033R0253', 'CA172033R0254', 'CA172033R0255', 'CA172033R0256', 'CA172033R0257', 'CA172033R0258', 'CA172033R0259', 'CA172033R0260', 'CA172033R0261', 'CA172033R0262', 'CA172033R0263', 'CA172033R0264', 'CA172033R0265', 'CA172033R0266', 'CA172033R0267', 'CA172033R0268', 'CA172033R0269', 'CA172033R0270', 'CA172033R0271', 'CA172033R0272', 'CA172033R0273', 'CA172033R0274', 'CA172033R0275', 'CA172133R0276', 'CA172133R0277', 'CA172133R0278', 'CA172133R0279', 'CA172133R0280', 'CA172133R0281', 'CA172133R0282', 'CA172133R0283', 'CA172133R0284', 'CA172133R0285', 'CA172133R0286', 'CA172133R0287', 'CA172133R0288', 'CA172133R0289', 'CA172233R0066', 'CA172233R0067', 'CA172233R0068', 'CA172233R0069', 'CA172233R0070', 'CA172233R0071', 'CA172233R0072', 'CA172233R0073', 'CA172233R0074', 'CA172233R0075', 'CA172233R0076', 'CA172233R0077', 'CA172233R0078', 'CA172233R0079', 'CA172233R0080', 'CA172233R0081', 'CA172233R0082', 'CA172233R0083', 'CA172233R0084', 'CA172233R0085', 'CA172233R0086', 'CA172233R0087', 'CA172233R0088', 'CA172233R0089', 'CA172233R0090', 'CA172233R0091', 'CA172233R0092', 'CA172333R0153', 'CA172333R0154', 'CA172333R0155', 'CA172333R0156', 'CA172333R0157', 'CA172333R0158', 'CA172333R0159', 'CA172333R0160', 'CA172333R0161', 'CA172333R0162', 'CA172333R0163', 'CA172333R0164', 'CA172333R0165', 'CA172333R0166', 'CA172333R0167', 'CA172333R0168', 'CA172333R0169', 'CA172333R0170', 'CA172333R0171', 'CA172333R0172', 'CA172333R0173', 'CA172333R0174', 'CA172333R0175', 'CA172333R0176', 'CA172333R0177', 'CA172333R0178', 'CA172333R0179', 'CA172433R0252', 'CA172433R0253', 'CA172433R0254', 'CA172433R0255', 'CA172433R0256', 'CA172433R0257', 'CA172433R0258', 'CA172433R0259', 'CA172433R0260', 'CA172433R0261', 'CA172433R0262', 'CA172433R0263', 'CA172433R0264', 'CA172433R0265', 'CA172433R0266', 'CA172433R0267', 'CA172433R0268', 'CA172433R0269', 'CA172433R0270', 'CA172433R0271', 'CA172433R0272', 'CA172433R0273', 'CA172433R0274', 'CA172533R0001', 'CA172533R0002', 'CA172533R0003', 'CA172533R0004', 'CA172533R0005', 'CA172533R0006', 'CA172533R0007', 'CA172533R0008', 'CA172533R0009', 'CA172533R0010', 'CA172533R0011', 'CA172533R0012', 'CA172533R0013', 'CA172533R0014', 'CA172533R0015', 'CA172533R0016', 'CA172533R0017', 'CA172533R0018', 'CA189333R0280', 'CA189333R0281', 'CA189333R0282', 'CA189333R0283', 'CA189333R0284', 'CA189333R0285', 'CA189333R0286', 'CA189333R0287', 'CA189333R0288', 'CA189333R0289', 'CA189333R0290', 'CA189333R0291', 'CA189333R0292', 'CA189333R0293', 'CA189333R0294', 'CA189333R0295', 'CA189333R0296', 'CA189333R0297', 'CA189333R0298', 'CA189333R0299', 'CA189333R0300', 'CA189333R0301', 'CA189333R0302', 'CA189333R0303', 'CA189333R0304', 'CA189333R0305', 'CA189333R0306', 'CA189333R0307', 'CA189333R0308', 'CA189333R0309', 'CA189333R0310', 'CA189333R0311', 'CA189333R0312', 'CA189333R0313', 'CA189333R0314', 'CA189333R0315', 'CA189333R0316', 'CA189333R0317', 'CA189833R0239', 'CA189833R0240', 'CA189833R0241', 'CA189833R0242', 'CA189833R0243', 'CA189833R0244', 'CA189833R0245', 'CA189833R0246', 'CA189833R0247', 'CA189833R0248', 'CA189833R0249', 'CA189833R0250', 'CA189833R0251', 'CA189833R0252', 'CA189833R0253', 'CA189833R0254', 'CA189833R0255', 'CA189833R0256', 'CA189833R0257', 'CA189833R0258', 'CA189833R0259', 'CA189833R0260', 'CA189833R0261', 'CA189833R0262', 'CA189833R0263', 'CA189833R0264', 'CA189833R0265', 'CA189833R0266', 'CA189833R0267', 'CA189833R0268', 'CA189833R0269', 'CA189833R0270', 'CA189833R0271', 'CA189833R0272', 'CA189833R0273', 'CA189833R0274', 'CA189833R0275', 'CA189833R0276', 'CA189833R0277', 'CA189833R0278', 'CA189833R0279', 'CA189833R0280', 'CA189833R0281', 'CA189833R0282', 'CA198233R0001', 'CA198233R0002', 'CA198233R0003', 'CA198233R0004', 'CA198233R0005', 'CA198233R0006', 'CA198233R0007', 'CA198233R0008', 'CA198233R0009', 'CA198233R0010', 'CA198233R0011', 'CA198233R0012', 'CA198233R0013', 'CA198233R0014', 'CA198233R0015', 'CA198233R0016', 'CA198233R0017', 'CA198233R0018', 'CA198233R0019', 'CA198233R0020', 'CA198233R0021', 'CA198233R0022', 'CA198233R0023', 'CA198233R0024', 'CA198233R0025', 'CA198233R0026', 'CA198233R0027', 'CA198233R0028', 'CA198233R0029', 'CA198233R0030', 'CA198233R0031', 'CA198233R0032', 'CA198233R0033', 'CA198233R0034', 'CA198233R0035', 'CA198233R0036', 'CA198233R0037', 'CA198233R0038', 'CA203933R0181', 'CA203933R0182', 'CA203933R0183', 'CA203933R0184', 'CA203933R0185', 'CA203933R0186', 'CA203933R0187', 'CA203933R0188', 'CA203933R0189', 'CA203933R0190', 'CA203933R0191', 'CA203933R0192', 'CA203933R0193', 'CA203933R0194', 'CA203933R0195', 'CA203933R0196', 'CA203933R0197', 'CA203933R0198', 'CA203933R0199', 'CA203933R0200', 'CA203933R0201', 'CA203933R0202', 'CA203933R0203', 'CA203933R0204', 'CA203933R0205', 'CA203933R0206', 'CA203933R0207', 'CA203933R0208', 'CA203933R0209', 'CA203933R0210', 'CA203933R0211', 'CA203933R0212', 'CA203933R0213', 'CA203933R0214', 'CA203933R0215', 'CA203933R0216', 'CA203933R0217', 'CA203933R0218', 'CA203933R0219', 'CA203933R0220', 'CA203933R0221', 'CA203933R0222', 'CA203933R0223', 'CA203933R0224', 'CA203933R0225', 'CA203933R0226', 'CA203933R0227', 'CA203933R0228', 'CA203933R0229', 'CA203933R0230', 'CA203933R0231', 'CA203933R0232', 'CA203933R0233', 'CA203933R0234', 'CA203933R0235', 'CA203933R0236', 'CA203933R0237', 'CA203933R0238', 'CA203933R0239', 'CA203933R0240', 'CA203933R0241', 'CA203933R0242', 'CA203933R0243', 'CA203933R0244', 'CA203933R0245', 'CA203933R0246', 'CA203933R0247', 'CA203933R0248', 'CA203933R0249', 'CA203933R0250', 'CA203933R0251', 'CA203933R0252', 'CA203933R0253', 'CA203933R0254', 'CA203933R0255', 'CA203933R0256', 'CA203933R0257', 'CA203933R0258', 'CA203933R0259', 'CA203933R0260', 'CA203933R0261', 'CA203933R0262', 'CA203933R0263', 'CA203933R0264', 'CA203933R0265', 'CA203933R0266', 'CA203933R0267', 'CA203933R0268', 'CA203933R0269', 'CA203933R0270', 'CA203933R0271', 'CA203933R0272', 'CA203933R0273', 'CA203933R0274', 'CA203933R0275', 'CA203933R0276', 'CA203933R0277', 'CA203933R0278', 'CA203933R0279', 'CA203933R0280', 'CA203933R0281', 'CA203933R0282', 'CA203933R0283', 'CA203933R0284', 'CA203933R0285', 'CA203933R0286', 'CA203933R0287', 'CA203933R0288', 'CA203933R0289', 'CA203933R0290', 'CA203933R0291', 'CA203933R0292', 'CA203933R0293', 'CA203933R0294', 'CA203933R0295', 'CA203933R0296', 'CA203933R0297', 'CA203933R0298', 'CA203933R0299', 'CA203933R0300', 'CA203933R0301', 'CA203933R0302', 'CA203933R0303', 'CA203933R0304', 'CA203933R0305', 'CA203933R0306', 'CA203933R0307', 'CA204033R0308', 'CA204033R0309', 'CA204033R0310', 'CA204033R0311', 'CA204033R0312', 'CA204033R0313', 'CA204033R0314', 'CA204033R0315', 'CA204033R0316', 'CA204033R0317', 'CA204033R0318', 'CA204033R0319', 'CA204033R0320', 'CA204033R0321', 'CA204033R0322', 'CA204033R0323', 'CA204033R0324', 'CA204033R0325', 'CA204033R0326', 'CA204033R0327', 'CA204033R0328', 'CA204033R0329', 'CA204033R0330', 'CA204033R0331', 'CA204033R0332', 'CA204033R0333', 'CA204033R0334', 'CA204033R0335', 'CA204033R0336', 'CA204033R0337', 'CA204033R0338', 'CA204033R0339', 'CA204033R0340', 'CA204033R0341', 'CA204033R0342', 'CA204033R0343', 'CA204033R0344', 'CA204033R0345', 'CA204033R0346', 'CA204033R0347', 'CA204033R0348', 'CA204033R0349', 'CA204033R0350', 'CA204033R0351', 'CA204033R0352', 'CA204033R0353', 'CA204033R0354', 'CA204033R0355', 'CA204033R0356', 'CA204033R0357', 'CA204033R0358', 'CA204033R0359', 'CA204033R0360', 'CA204033R0361', 'CA204033R0362', 'CA204033R0363', 'CA204033R0364', 'CA204033R0365', 'CA204033R0366', 'CA204033R0367', 'CA204033R0368', 'CA204033R0369', 'CA204033R0370', 'CA204033R0371', 'CA204033R0372', 'CA204033R0373', 'CA204033R0374', 'CA204033R0375', 'CA204033R0376', 'CA204033R0377', 'CA204033R0378', 'CA204033R0379', 'CA204033R0380', 'CA204033R0381', 'CA204033R0382', 'CA204033R0383', 'CA204033R0384', 'CA204033R0385', 'CA204033R0386', 'CA204033R0387', 'CA204033R0388', 'CA204033R0389', 'CA204033R0390', 'CA204033R0391', 'CA204033R0392', 'CA204033R0393', 'CA204033R0394', 'CA204033R0395', 'CA204033R0396', 'CA204033R0397', 'CA204033R0398', 'CA204033R0399', 'CA204033R0400', 'CA204033R0401', 'CA204033R0402', 'CA204033R0403', 'CA204033R0404', 'CA204033R0405', 'CA204033R0406', 'CA204033R0407', 'CA204033R0408', 'CA204033R0409', 'CA204033R0410', 'CA204033R0411', 'CA204033R0412', 'CA204033R0413', 'CA204033R0414', 'CA204033R0415', 'CA204033R0416', 'CA204033R0417', 'CA204033R0418', 'CA204033R0419', 'CA204033R0420', 'CA204133R0035', 'CA204133R0036', 'CA204133R0037', 'CA204133R0038', 'CA204133R0039', 'CA204133R0040', 'CA204133R0041', 'CA204133R0042', 'CA204133R0043', 'CA204133R0044', 'CA204133R0045', 'CA204133R0046', 'CA204133R0047', 'CA204133R0048', 'CA204133R0049', 'CA204133R0050', 'CA204133R0051', 'CA204133R0052', 'CA204133R0053', 'CA204133R0054', 'CA204133R0055', 'CA204133R0056', 'CA204133R0057', 'CA204133R0058', 'CA204133R0059', 'CA204133R0060', 'CA204133R0061', 'CA204133R0062', 'CA204133R0063', 'CA204133R0064', 'CA204133R0065', 'CA204133R0066', 'CA204133R0067', 'CA204133R0068', 'CA204133R0069', 'CA204133R0070', 'CA204133R0071', 'CA204133R0072', 'CA204133R0073', 'CA204133R0074', 'CA204133R0075', 'CA204133R0076', 'CA204133R0077', 'CA204133R0078', 'CA204133R0079', 'CA204133R0080', 'CA204133R0081', 'CA204133R0082', 'CA204133R0083', 'CA204133R0084', 'CA204133R0085', 'CA204133R0086', 'CA204133R0087', 'CA204133R0088', 'CA204133R0089', 'CA204133R0090', 'CA204133R0091', 'CA204133R0092', 'CA204133R0093', 'CA204133R0094', 'CA204133R0095', 'CA204133R0096', 'CA204133R0097', 'CA204133R0098', 'CA204133R0099', 'CA204133R0100', 'CA204133R0101', 'CA204133R0102', 'CA204133R0103', 'CA204133R0104', 'CA204133R0105', 'CA204133R0106', 'CA204133R0107', 'CA204133R0108', 'CA204133R0109', 'CA204133R0110', 'CA204133R0111', 'CA204133R0112', 'CA204133R0113', 'CA204133R0114', 'CA204133R0115', 'CA204233R0146', 'CA204233R0147', 'CA204233R0148', 'CA204233R0149', 'CA204233R0150', 'CA204233R0151', 'CA204233R0152', 'CA204233R0153', 'CA204233R0154', 'CA204233R0155', 'CA204233R0156', 'CA204233R0157', 'CA204233R0158', 'CA204233R0159', 'CA204233R0160', 'CA204233R0161', 'CA204233R0162', 'CA204233R0163', 'CA204233R0164', 'CA204233R0165', 'CA204233R0166', 'CA204233R0167', 'CA204233R0168', 'CA204233R0169', 'CA204233R0170', 'CA204233R0171', 'CA204233R0172', 'CA204233R0173', 'CA204233R0174', 'CA204233R0175', 'CA204333R0238', 'CA204333R0239', 'CA204333R0240', 'CA204333R0241', 'CA204333R0242', 'CA204333R0243', 'CA204333R0244', 'CA204333R0245', 'CA204333R0246', 'CA204333R0247', 'CA204333R0248', 'CA204333R0249', 'CA204333R0250', 'CA204333R0251', 'CA204333R0252', 'CA204333R0253', 'CA204333R0254', 'CA512033R0002', 'CA512033R0003', 'CA512033R0004', 'CA512033R0005', 'CA512033R0006', 'CA512033R0007', 'CA512033R0008', 'CA512033R0009', 'CA512033R0010', 'CA512033R0011', 'CA512033R0012', 'CA512033R0013', 'CA512033R0014', 'CA512033R0015', 'CA512033R0016', 'CA512033R0017', 'CA512033R0018', 'CA512033R0019', 'CA512033R0020', 'CA512033R0021', 'CA512033R0022', 'CA512033R0023', 'CA512033R0024', 'CA512033R0025', 'CA512033R0026', 'CA512033R0027', 'CA512033R0028', 'CA512233R0001', 'CA512233R0002', 'CA512233R0003', 'CA512233R0004', 'CA512233R0005', 'CA512233R0006', 'CA512233R0007', 'CA512233R0008', 'CA512233R0009', 'CA512233R0010', 'CA512233R0011', 'CA512233R0012', 'CA512233R0013', 'CA512233R0014', 'CA512233R0015', 'CA512333R0001', 'CA512333R0002', 'CA512333R0003', 'CA512333R0004', 'CA512333R0005', 'CA512333R0006', 'CA512333R0007', 'CA512333R0008', 'CA512333R0009', 'CA512333R0010', 'CA512333R0011', 'CA512333R0012', 'CA512333R0013', 'CA512333R0014', 'CA512333R0015', 'CA512333R0016', 'CA512333R0017', 'CA512333R0018', 'CA512333R0019', 'CA512333R0020', 'CA512333R0021', 'CA512333R0022', 'CA512333R0023', 'CA512333R0024', 'CA512333R0025', 'CA512333R0026', 'CA512333R0027', 'CA512333R0028', 'CA512333R0029', 'CA512333R0030', 'CA512333R0031', 'CA512333R0032', 'CA512333R0033', 'CA512333R0034', 'CA512333R0035', 'CA512333R0036', 'CA512333R0037', 'CA512333R0038', 'CA512333R0039', 'CA512333R0040', 'CA512333R0041', 'CA512333R0042', 'CA512333R0043', 'CA512333R0044', 'CA512333R0045', 'CA512333R0046', 'CA512333R0047', 'CA512333R0048', 'CA512333R0049', 'CA512333R0050', 'CA512333R0051', 'CA512333R0052', 'CA512333R0053', 'CA512333R0054', 'CA512333R0055', 'CA512333R0056', 'CA512333R0057', 'CA512333R0058', 'CA512333R0059', 'CA512333R0060', 'CA512333R0061', 'CA512333R0062', 'CA512333R0063', 'CA512333R0064', 'CA512333R0065', 'CA512333R0066', 'CA512333R0067', 'CA512333R0068', 'CA512333R0069', 'CA512333R0070', 'CA512333R0071', 'CA512333R0072', 'CA512333R0073', 'CA512333R0074', 'CA512333R0075', 'CA512333R0076', 'CA512333R0077', 'CA512333R0078', 'CA512333R0079', 'CA512333R0080', 'CA512333R0081', 'CA512333R0082', 'CA512333R0083', 'CA512333R0084', 'CA512333R0085', 'CA512333R0086', 'CA512333R0087', 'CA512333R0088', 'CA512333R0089', 'CA512333R0090', 'CA512333R0091', 'CA512333R0092', 'CA512333R0093', 'CA512333R0094', 'CA512333R0095', 'CA512333R0096', 'CA512333R0097', 'CA512333R0098', 'CA512333R0099', 'CA512333R0100', 'CA512333R0101', 'CA512333R0102', 'CA512333R0103', 'CA512333R0104', 'CA512333R0105', 'CA512333R0106', 'CA512333R0107', 'CA512333R0108', 'CA512333R0109', 'CA512333R0110', 'CA512333R0111', 'CA512333R0112', 'CA512333R0113', 'CA512333R0114', 'CA512333R0115', 'CA512333R0116', 'CA512333R0117', 'CA512333R0118', 'CA512333R0119', 'CA512333R0120', 'CA512333R0121', 'CA512333R0122', 'CA512333R0123', 'CA512333R0124', 'CA512333R0125', 'CA512333R0126', 'CA512333R0127', 'CA512333R0128', 'CA512333R0129', 'CA512333R0130', 'CA512333R0131', 'CA512333R0132', 'CA512333R0133', 'CA512333R0134', 'CA512333R0135', 'CA512333R0136', 'CA512333R0137', 'CA512333R0138', 'CA512333R0139', 'CA512333R0140', 'CA512333R0141', 'CA512333R0142', 'CA512333R0143', 'CA512433R0012', 'CA512433R0013', 'CA512433R0014', 'CA512433R0015', 'CA512433R0016', 'CA512433R0017', 'CA512433R0018', 'CA512433R0019', 'CA512433R0020', 'CA512433R0021', 'CA512433R0022', 'CA512433R0023', 'CA512433R0024', 'CA512433R0025', 'CA512433R0026', 'CA512433R0027', 'CA512433R0028', 'CA512433R0029', 'CA512433R0030', 'CA512433R0031', 'CA512433R0032', 'CA512433R0033', 'CA512433R0034', 'CA512433R0035', 'CA512433R0036', 'CA512433R0037', 'CA512433R0038', 'CA512433R0039', 'CA512433R0040', 'CA512433R0041', 'CA512433R0042', 'CA512433R0043', 'CA512533R0044', 'CA512533R0045', 'CA512533R0046', 'CA512533R0047', 'CA512533R0048', 'CA512533R0049', 'CA512533R0050', 'CA512533R0051', 'CA512533R0052', 'CA512533R0053', 'CA512533R0054', 'CA512533R0055', 'CA512533R0056', 'CA512533R0057', 'CA512533R0058', 'CA512633R0095', 'CA512633R0094', 'CA512633R0093', 'CA512633R0092', 'CA512633R0091', 'CA512633R0090', 'CA512633R0089', 'CA512633R0088', 'CA512633R0087', 'CA512633R0086', 'CA512633R0085', 'CA512633R0084', 'CA512633R0083', 'CA512633R0082', 'CA512633R0081', 'CA512633R0080', 'CA512633R0079', 'CA512633R0078', 'CA512633R0077', 'CA512633R0076', 'CA512733R0096', 'CA512733R0097', 'CA512733R0098', 'CA512733R0099', 'CA512733R0100', 'CA512733R0101', 'CA512733R0102', 'CA512733R0103', 'CA512733R0104', 'CA512733R0105', 'CA512733R0106', 'CA512733R0107', 'CA512733R0108', 'CA512733R0109', 'CA512733R0110', 'CA512733R0111', 'CA512733R0112', 'CA512733R0113', 'CA512733R0114', 'CA512733R0115', 'CA512733R0116', 'CA512733R0117', 'CA512733R0118', 'CA512733R0119', 'CA512733R0120', 'CA512733R0121', 'CA512733R0122', 'CA512733R0123', 'CA512733R0124', 'CA512733R0125', 'CA512733R0126', 'CA512733R0127', 'CA512733R0128', 'CA512733R0129', 'CA512733R0130', 'CA512733R0131', 'CA512733R0132', 'CA512733R0133', 'CA512733R0134', 'CA512733R0135', 'CA512733R0136']

    sql_string = "SELECT image_id FROM images"
    data = ctd.get_data_from_db(sql_string)
    img_ids = data.values.tolist()
    img_ids = [item for sublist in img_ids for item in sublist]

    import random
    random.shuffle(img_ids)

    extract_for_all_images(img_ids, overwrite=False, catch=True, verbose=True)
