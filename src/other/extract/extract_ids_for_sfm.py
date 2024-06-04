import src.base.connect_to_database as ctd

USE_SHP_HEIGHT = True


def extract_ids_for_sfm(min_nr, image_xml, resampled, min_complexity=0,
                        position=True, height=True, focal_length=True,
                        maximum_id_distance=1):

    if maximum_id_distance < 1:
        raise ValueError("The maximum ID distance must be at least 1.")

    conn = ctd.establish_connection()

    sql_string = "SELECT images.image_id, images.tma_number, " \
                 "images_extracted.complexity, " \
                 "images_file_paths.path_xml_file, " \
                 "images_file_paths.path_downloaded_resampled, " \
                 "images_extracted.position_exact, " \
                 "images_extracted.focal_length, images_extracted.height " \
                 "FROM images JOIN images_extracted " \
                 "on images.image_id = images_extracted.image_id " \
                 "JOIN images_file_paths ON " \
                 "images.image_id = images_file_paths.image_id"
    data = ctd.execute_sql(sql_string, conn)

    # get the height data from shp file
    if USE_SHP_HEIGHT:
        sql_string = "SELECT image_id, altitude FROM images"
        data_height = ctd.execute_sql(sql_string, conn)

        # merge the height data with the original data
        data = data.merge(data_height, on='image_id', how='left')

        # remove values with -99999
        data = data[data['altitude'] != -99999]

        # put values from altitude to height if height is null
        data['height'] = data['height'].fillna(data['altitude'])

        # drop the altitude column
        data = data.drop(columns=['altitude'])

    print("Original images: ", data.shape[0])

    # filer based on image_xml
    if image_xml:
        data = data[data['path_xml_file'].notnull()]

        print("Images with xml: ", data.shape[0])

    # filter based on position
    if position:
        data = data[data['position_exact'].notnull()]

        print("Images with position: ", data.shape[0])

    # filter based on height
    if height:
        data = data[data['height'].notnull()]

        print("Images with height: ", data.shape[0])

    # filter based on focal length
    if focal_length:
        data = data[data['focal_length'].notnull()]

        print("Images with focal length: ", data.shape[0])

    # filter based on complexity
    if min_complexity > 0:
        data = data[data['complexity'] >= min_complexity]
        print("Images with complexity: ", data.shape[0])

    # filter based on resampled
    if resampled:
        data = data[data['path_file_resampled'].notnull()]

        print("Images with resampled: ", data.shape[0])

    # Filter to include only those groups where 'tma_number' appears at least 'min_nr' times
    data = data.groupby('tma_number').filter(lambda x: len(x) >= min_nr)

    # Grouping image IDs by 'tma_number' into lists and sorting those lists
    grouped_ids = data.groupby('tma_number')['image_id'].apply(lambda x: sorted(x.tolist()))

    # Convert keys to strings
    grouped_ids = grouped_ids.reset_index().set_index('tma_number')['image_id'].to_dict()
    grouped_ids = {str(int(k)): v for k, v in grouped_ids.items()}

    # Initialize the final dict
    final_groups = {}

    # check groups for the minimum distance between images
    for key, ids in grouped_ids.items():
        current_group = []
        last_id = None
        group_count = 1

        for image_id in ids:
            # Extract the last four digits of the image ID
            current_id = int(image_id[-4:])

            # If this is the first ID in the group, initialize last_id
            if last_id is None:
                last_id = current_id

            # Calculate the distance between current and last ID
            distance = current_id - last_id

            # Check if the distance exceeds the maximum allowed distance
            if distance > maximum_id_distance:
                # Check if the current group meets the minimum number of images
                if len(current_group) >= min_nr:
                    # Add the current group to the new data with a new key
                    new_key = f"{key}_{group_count}"
                    final_groups[new_key] = current_group
                    group_count += 1
                # Reset the current group
                current_group = []

            # Add the current ID to the group and update last_id
            current_group.append(image_id)
            last_id = current_id

        # Check the last group if it meets the minimum number of images
        if len(current_group) >= min_nr:
            new_key = f"{key}_{group_count}"
            final_groups[new_key] = current_group


    return final_groups