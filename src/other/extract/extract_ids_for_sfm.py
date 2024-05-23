import src.base.connect_to_database as ctd


def extract_ids_for_sfm(min_nr, image_xml=True,
                        position=True, height=True,
                        focal_length=True, resampled=True):

    conn = ctd.establish_connection()

    sql_string = "SELECT images.image_id, images.tma_number, " \
                 "images.path_file_resampled, " \
                 "images_extracted.position_exact, " \
                 "images_extracted.focal_length, images_extracted.height " \
                 "FROM images JOIN images_extracted " \
                 "on images.image_id = images_extracted.image_id "
    data = ctd.execute_sql(sql_string, conn)

    print("Original images: ", data.shape[0])

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

    # filter based on resampled
    if resampled:
        data = data[data['path_file_resampled'].notnull()]

        print("Images with resampled: ", data.shape[0])

    # Filter to include only those groups where 'tma_number' appears at least 'min_nr' times
    data = data.groupby('tma_number').filter(lambda x: len(x) >= min_nr)

    # Grouping image IDs by 'tma_number' into lists and sorting those lists
    grouped_ids = data.groupby('tma_number')['image_id'].apply(lambda x: sorted(x.tolist()))

    # Return a dictionary with 'tma_number' as keys and sorted lists of 'image_id' as values
    return grouped_ids.to_dict()