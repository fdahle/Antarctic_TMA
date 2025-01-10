
mode = "all" # can be 'all', 'flight_paths' or 'area

import src.base.connect_to_database as ctd
conn = ctd.establish_connection()

if mode == "all":
    image_ids = None
elif mode == "flight_paths":
    flight_paths = ['2152', '1684', '1820', '2157']
    import src.other.extract.extract_ids_by_flightpath as eibf
    image_ids = eibf.extract_ids_by_area(flight_paths, conn=conn)
elif mode == "area":
    raise NotImplementedError

# get extracted data for images with this flight paths
sql_string = f"SELECT * FROM images_extracted"
if image_ids is not None:
    sql_string += f" WHERE image_id IN {tuple(image_ids)}"
data_extracted = ctd.execute_sql(sql_string, conn)

# get fid-mark data for these images
sql_string = f"SELECT * FROM images_fid_points"
if image_ids is not None:
    sql_string += f" WHERE image_id IN {tuple(image_ids)}"
data_fid = ctd.execute_sql(sql_string, conn)

# merge data based on image_id
data = data_extracted.merge(data_fid, on='image_id')

# stats for subsets
for subset_key in ["n", "e", "s", "w"]:
    # get number of non-null values
    subsets = data[f'subset_{subset_key}_x']
    subsets = subsets[~subsets.isnull()]
    subset_count = subsets.shape[0]
    missing_count = data.shape[0] - subset_count

    # get number of estimated fid marks (=True)
    subsets_estimated = data[f'subset_{subset_key}_estimated']
    subsets_estimated = subsets_estimated[subsets_estimated == True]
    subsets_estimated_count = subsets_estimated.shape[0]

    print(f"Subset {subset_key} (ex:{subset_count}|est:{subsets_estimated_count}|"
          f"mi:{missing_count}|to:{data.shape[0]})")

# stats for fid marks
for fid_key in ["1", "2", "3", "4", "5", "6", "7", "8"]:
    # get number of non-null values
    fid_marks = data[f'fid_mark_{fid_key}_x']
    fid_marks = fid_marks[~fid_marks.isnull()]
    fid_marks_count = fid_marks.shape[0]
    missing_count = data.shape[0] - fid_marks_count

    # get number of estimated fid marks (=True)
    fid_marks_estimated = data[f'fid_mark_{fid_key}_estimated']
    fid_marks_estimated = fid_marks_estimated[fid_marks_estimated == True]
    fid_marks_estimated_count = fid_marks_estimated.shape[0]

    print(f"Fid mark {fid_key} (ex:{fid_marks_count}|est:{fid_marks_estimated_count}|"
          f"mi:{missing_count}|to:{data.shape[0]})")

# stats for height
heights = data['height']
heights = heights[~heights.isnull()]
heights_count = heights.shape[0]
missing_count = data.shape[0] - heights_count

height_estimated = data['height_estimated']
height_estimated = height_estimated[height_estimated == True]
height_estimated_count = height_estimated.shape[0]

print(f"Height (ex:{heights_count}|est:{height_estimated_count}|"
      f"mi:{missing_count}|to:{data.shape[0]})")

# stats for focal length
focal_lengths = data['focal_length']
focal_lengths = focal_lengths[~focal_lengths.isnull()]
focal_lengths_count = focal_lengths.shape[0]
missing_count = data.shape[0] - focal_lengths_count

focal_lengths_estimated = data['focal_length_estimated']
focal_lengths_estimated = focal_lengths_estimated[focal_lengths_estimated == True]
focal_lengths_estimated_count = focal_lengths_estimated.shape[0]

print(f"Focal length (ex:{focal_lengths_count}|est:{focal_lengths_estimated_count}|"
      f"mi:{data.shape[0]}|to:{data.shape[0]})")

SHOW_RANDOM_MISSING = True
missing_col = "subset_n_x"
if SHOW_RANDOM_MISSING:

    # get a random image id with missing data
    missing_data = data[data[missing_col].isnull()]
    random_missing = missing_data.sample(1)

    print(random_missing)

    # get the image id
    image_id = random_missing['image_id'].values[0]

    # load the image
    import src.load.load_image as li
    image = li.load_image(image_id)

    # display the image
    import src.display.display_images as di
    di.display_images([image])

