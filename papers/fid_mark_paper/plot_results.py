mode = "flight_paths"  # Can be 'all', 'flight_paths' or 'area'

import src.base.connect_to_database as ctd
conn = ctd.establish_connection()

# LOOK FOR FLIGHT PATHS WHERE HEIGHT IS INTEGRATED
if mode == "all":
    image_ids = None
elif mode == "flight_paths":
    # Ten longest flight paths on the AP
    flight_paths_10 = ["1821", "1816", "1833", "2137", "1825", "2136",
                       "2143", "1826", "1813", "2141"]
    flight_paths_20 = ["2140", "2073", "1827", "1684", "2142",
                       "1824", "1846", "2139", "2075"]

    flight_paths = flight_paths_10 + flight_paths_20

    import src.other.extract.extract_ids_by_flightpath as eibf
    image_ids = eibf.extract_ids_by_area(flight_paths, conn=conn)
elif mode == "area":
    raise NotImplementedError

# Get extracted data for images with these flight paths
sql_string = f"SELECT * FROM images_extracted"
if image_ids is not None:
    sql_string += f" WHERE image_id IN {tuple(image_ids)}"
data_extracted = ctd.execute_sql(sql_string, conn)

# Get fid-mark data for these images
sql_string = f"SELECT * FROM images_fid_points"
if image_ids is not None:
    sql_string += f" WHERE image_id IN {tuple(image_ids)}"
data_fid = ctd.execute_sql(sql_string, conn)

# Merge data based on image_id
data = data_extracted.merge(data_fid, on='image_id')

# Helper function to count extracted and estimated values
def count_extracted_and_estimated(data_col, estimated_col):
    extracted = data_col[~data_col.isnull() & (estimated_col == False)].shape[0]
    estimated = data_col[~data_col.isnull() & (estimated_col == True)].shape[0]
    return extracted, estimated

# Stats for subsets
for subset_key in ["n", "e", "s", "w"]:
    data_col = data[f'subset_{subset_key}_x']
    estimated_col = data[f'subset_{subset_key}_estimated']

    subset_extracted_count, subset_estimated_count = count_extracted_and_estimated(data_col, estimated_col)
    missing_count = data.shape[0] - subset_extracted_count - subset_estimated_count

    print(f"Subset {subset_key} (extracted:{subset_extracted_count}|estimated:{subset_estimated_count}|"
          f"missing:{missing_count}|total:{data.shape[0]})")

# Stats for fid marks
for fid_key in ["1", "2", "3", "4", "5", "6", "7", "8"]:
    data_col = data[f'fid_mark_{fid_key}_x']
    estimated_col = data[f'fid_mark_{fid_key}_estimated']

    fid_extracted_count, fid_estimated_count = count_extracted_and_estimated(data_col, estimated_col)
    missing_count = data.shape[0] - fid_extracted_count - fid_estimated_count

    print(f"Fid mark {fid_key} (extracted:{fid_extracted_count}|estimated:{fid_estimated_count}|"
          f"missing:{missing_count}|total:{data.shape[0]})")

# Stats for altitude
altitude_extracted_count, altitude_estimated_count = count_extracted_and_estimated(
    data['altimeter_value'], data['altimeter_estimated']
)
missing_count = data.shape[0] - altitude_extracted_count - altitude_estimated_count

print(f"Altitude (extracted:{altitude_extracted_count}|estimated:{altitude_estimated_count}|"
      f"missing:{missing_count}|total:{data.shape[0]})")

# Stats for height
height_extracted_count, height_estimated_count = count_extracted_and_estimated(
    data['height'], data['height_estimated']
)
missing_count = data.shape[0] - height_extracted_count - height_estimated_count

print(f"Height (extracted:{height_extracted_count}|estimated:{height_estimated_count}|"
      f"missing:{missing_count}|total:{data.shape[0]})")

# Stats for focal length
focal_extracted_count, focal_estimated_count = count_extracted_and_estimated(
    data['focal_length'], data['focal_length_estimated']
)
missing_count = data.shape[0] - focal_extracted_count - focal_estimated_count

print(f"Focal length (extracted:{focal_extracted_count}|estimated:{focal_estimated_count}|"
      f"missing:{missing_count}|total:{data.shape[0]})")

# Random missing data check
SHOW_RANDOM_MISSING = False
SHOW_RANDOM_INCLUDING = True
INCLUDE_ESTIMATED = False
col_nm = "altimeter"

if SHOW_RANDOM_MISSING:
    i = 0

    while i < 100:
        i += 1

        # Get a random image id with missing data
        if INCLUDE_ESTIMATED:
            missing_data = data[data[col_nm].isnull() & data[f'{col_nm}_estimated'] == False]
        else:
            missing_data = data[data[col_nm].isnull()]

        if missing_data.shape[0] == 0:
            print("No missing data")
            exit()

        random_missing = missing_data.sample(1)
        print(random_missing)

        # Get the image id
        image_id = random_missing['image_id'].values[0]

        # Load the image
        import src.load.load_image as li
        try:
            image = li.load_image(image_id)
        except:
            continue

        # Display the image
        import src.display.display_images as di
        di.display_images([image])

if SHOW_RANDOM_INCLUDING:
    i = 0

    if col_nm.startswith("fid_mark"):
        key = f"{col_nm}_x"
    elif col_nm == "altimeter":
        key = "altimeter_value"
    else:
        key = col_nm

    while i < 100:
        i += 1

        # Get a random image id with missing data
        if INCLUDE_ESTIMATED is False:
            existing_data = data[~data[key].isnull() & data[f'{col_nm}_estimated'] == False]
        else:
            existing_data = data[~data[key].isnull()]

        if existing_data.shape[0] == 0:
            print("No existing data")
            exit()

        random_existing = existing_data.sample(1)

        # Get the image id
        image_id = random_existing['image_id'].values[0]

        print(image_id, random_existing[key].values[0])

        # Load the image
        import src.load.load_image as li
        try:
            image = li.load_image(image_id)
        except:
            continue

        # Display the image
        import src.display.display_images as di

        # depending on the column name, we need different additional attributes
        if "fid_mark" in col_nm:

            # get x and y values
            x = random_existing[f'{col_nm}_x'].values[0]
            y = random_existing[f'{col_nm}_y'].values[0]

            di.display_images([image], points=[[(x, y)]])

        if "focal_length" in col_nm:
            # get text boxes for text
            text_boxes = random_existing['text_bbox'].values[0]
            # remove all '(' and ')'
            text_boxes = text_boxes.replace("(", "").replace(")", "")
            # split boxes
            text_boxes = text_boxes.split(";")
            # convert to list of tuples
            text_boxes = [tuple(map(int, box.split(","))) for box in text_boxes]

            from shapely.geometry import Polygon

            polygons = [
                Polygon([(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)])
                for (x_min, y_min, x_max, y_max) in text_boxes
            ]

            print(text_boxes)
            di.display_images([image], polygons=[polygons])

