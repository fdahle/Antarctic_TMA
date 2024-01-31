import base.connect_to_db as ctd
import base.load_image_from_file as liff


def train_subset_extractor(num_images=250, model_name="model", model_path=None):

    """
    train_subset_extractor(num_images):
    This function trains the detector for the subsets and saves it to the folder. From the database a certain amount of
    random example images is loaded together with the subset coordinates for the training
    Args:
        - num_images (int, 250): With how many examples should be trained
        - model_name(str, "model_"): What is the name of the model (with "N", "E", "S", "W" at the end attached)
        - model_path(str, None): Where should the model be stored. If none the default path is used
    Returns:
        None
    """

    directions = ["n", "s", "e", "w"]

    if model_path is None:
        model_path = "/data_1/ATM/data/machine_learning/subsets"

    for direction in directions:

        # get all subset coordinates
        sql_string = f"SELECT image_id, subset_{direction}_x, subset_{direction}_y, subset_width, subset_height " \
                     f"from images_properties WHERE " \
                     f"subset_{direction}_x IS NOT NULL and subset_{direction}_y IS NOT NULL"

        data = ctd.get_data_from_db(sql_string, catch=False)

        crops = []
        boxes = []

        # shuffle the data
        data = data.sample(frac=1).reset_index(drop=True)

        for index, row in data.iterrows():

            print(f"Load {row['image_id']}")

            if index == num_images:
                break

            image = liff.load_image_from_file(row["image_id"])

            min_y = int(row[f"subset_{direction}_y"])
            max_y = int(row[f"subset_{direction}_y"] + row["subset_height"])
            min_x = int(row[f"subset_{direction}_x"])
            max_x = int(row[f"subset_{direction}_x"] + row["subset_width"])

            if min_y < 0:
                print("WARNING")
                min_y = 0
            if min_x < 0:
                print("WARNING")
                min_x = 0

            subset_factor = 0.1

            # get size params of the image
            height, width = image.shape
            mid_y = int(height / 2)
            mid_x = int(width / 2)
            subset_height = int(subset_factor * height)
            subset_width = int(subset_factor * width)

            # TODO workaround
            if direction == "n" and min_y > 1000:
                continue
            elif direction == "s" and min_y < 8750:
                continue

            # init crop so that ide is not complaining
            crop = None

            if direction == "n":
                crop = image[0:subset_height, mid_x - subset_width:mid_x + subset_width]
                min_x = min_x - (mid_x - subset_width)
                max_x = max_x - (mid_x - subset_width)
            elif direction == "e":
                crop = image[mid_y - subset_height:mid_y + subset_height, width - subset_width:width]
                min_x = min_x - (width-subset_width)
                max_x = max_x - (width-subset_width)
                min_y = min_y - (mid_y - subset_height)
                max_y = max_y - (mid_y - subset_height)
            elif direction == "s":
                crop = image[height - subset_height:height, mid_x - subset_width:mid_x + subset_width]
                min_x = min_x - (mid_x - subset_width)
                max_x = max_x - (mid_x - subset_width)
                min_y = min_y - (height - subset_height)
                max_y = max_y - (height - subset_height)
            elif direction == "w":
                crop = image[mid_y - subset_height:mid_y + subset_height, 0:subset_width]
                min_y = min_y - (mid_y - subset_height)
                max_y = max_y - (mid_y - subset_height)

            crops.append(crop)

            box = [dlib.rectangle(left=min_x, top=min_y, right=max_x, bottom=max_y)]
            boxes.append(box)

        print(f"Train detector for {direction}")

        # set options for dlib
        options = dlib.simple_object_detector_training_options()
        options.add_left_right_image_flips = False
        options.C = 5
        options.num_threads = 10
        options.be_verbose = True

        detector = dlib.train_simple_object_detector(crops, boxes, options)
        file_name = model_path + "/" + model_name + "_" + direction + ".svm"

        detector.save(file_name)
