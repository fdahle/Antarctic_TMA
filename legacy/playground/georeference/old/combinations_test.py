# get the width and height of the satellite image
sat_width = 10304
sat_height = 10341

# set the params
overlap = 1 / 3  # how much overlap do we want to have
max_order = 3  # what is the maximum order we're checking
success_criteria = "num_points"  # What is a success? 'num_points' or 'avg_conf'

# calculate the step size
step_x = int((1 - overlap) * sat_width)
step_y = int((1 - overlap) * sat_height)

# track what we did check already
lst_checked_combinations = []

# define in which rotations we want to check
rotations = [0, 90, 180, 270]

# iterate the orders (start with the lowest first)
for order in range(max_order):

    # create a combinations dict
    combinations = []
    for i in range(-order * step_x, (order + 1) * step_x, step_x):
        for j in range(-order * step_y, (order + 1) * step_y, step_y):
            combinations.append([i, j])

    # per combination, we want to check our performance
    best_combination = None
    best_nr_of_points = 0
    best_avg_conf = 0

    # check all combinations
    for combination in combinations:

        # copy the original sat bounds (to not change them)
        adapted_sat_bounds = copy.deepcopy(sat_bounds)

        # first adapt x
        adapted_sat_bounds[0] = adapted_sat_bounds[0] + combination[0]
        adapted_sat_bounds[2] = adapted_sat_bounds[2] + combination[0]

        # and then y
        adapted_sat_bounds[1] = adapted_sat_bounds[1] + combination[1]
        adapted_sat_bounds[3] = adapted_sat_bounds[3] + combination[1]

        sat_image, sat_transform = lsd.load_satellite_data(adapted_sat_bounds,
                                                           return_transform=True,
                                                           catch=catch, verbose=False)

        # another loop to check all 4 rotations
        for rotation in rotations:

            # copy the image to not change it and then rotate it
            img_rotated = copy.deepcopy(img_adjusted)
            img_rotated = ri.rotate_image(img_rotated, rotation)

            # find the tie-points
            points_all, conf_all = ftp.find_tie_points(sat_image, img_rotated,
                                                       mask_1=None, mask_2=None,
                                                       min_threshold=0.2,
                                                       extra_mask_padding=10,
                                                       additional_matching=True,
                                                       extra_matching=True,
                                                       keep_resized_points=True,
                                                       keep_additional_points=True,
                                                       catch=catch, verbose=False, pbar=pbar)

            # get the average conf
            if conf_all is None:
                avg_conf = 0
            else:
                avg_conf = np.mean(np.asarray(conf_all))

            if success_criteria == "num_points" and points_all.shape[0] > best_nr_of_points:
                best_combination = str(combination) + ";" + rotation
                best_nr_of_points = points_all.shape[0]
                best_avg_conf = avg_conf
            elif success_criteria == "avg_conf" and avg_conf > best_avg_conf:
                best_combination = str(combination) + ";" + rotation
                best_nr_of_points = points_all.shape[0]
                best_avg_conf = avg_conf

    # do we need another order?
    if best_nr_of_points > 25:
        break



    print(best_combination, best_nr_of_points, best_combination)