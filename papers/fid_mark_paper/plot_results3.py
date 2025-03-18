import pandas as pd
import src.base.connect_to_database as ctd

# Establish database connection
conn = ctd.establish_connection()

# Define flight paths
flight_paths_10 = ["1821", "1816", "1833", "2137", "1825", "2136",
                   "2143", "1826", "1813", "2141"]
flight_paths_20 = ["2140", "2073", "1822", "1827", "1684", "2142",
                   "1824", "1846", "2139", "2075"]
flight_paths = flight_paths_10 + flight_paths_20

import src.other.extract.extract_ids_by_flightpath as eibf

image_ids = eibf.extract_ids_by_area(flight_paths, conn=conn)

# Define relevant columns
fid_mark_cols = [f"fid_mark_{i}_x" for i in range(5, 9)]
estimated_cols = [f"fid_mark_{i}_estimated" for i in range(5, 9)]

# Query fid mark data
sql_string = "SELECT image_id, " + ", ".join(fid_mark_cols + estimated_cols) + " FROM images_fid_points"
if image_ids is not None:
    sql_string += f" WHERE image_id IN {tuple(image_ids)}"

data_fid = ctd.execute_sql(sql_string, conn)

# Extract flight path and camera direction
data_fid['flight_path'] = data_fid['image_id'].str[2:6]
data_fid['camera_direction'] = data_fid['image_id'].str[8:9]

# Initialize an empty DataFrame
fid_counts = None

# Get the total count of images per flight path and camera direction
count_all = data_fid.groupby(['flight_path', 'camera_direction']).size().reset_index(name='total_images')

check_type="fid_mark"
check_type="subset"

# Iterate over fid_mark positions and compute extracted fid marks
for i in range(5, 9):
    # Filter out estimated fid marks
    data_i_fid = data_fid[data_fid[f"fid_mark_{i}_estimated"] == False]

    # Count the number of fid marks per flight path and camera direction
    count_extracted = data_i_fid.groupby(['flight_path', 'camera_direction']).size().reset_index(name=f'fid_{i}')

    # Merge with total image count
    count_final = count_all.merge(count_extracted, on=['flight_path', 'camera_direction'], how='left').fillna(0)
    count_final[f'fid_{i}'] = count_final['total_images'] - count_final[f'fid_{i}']

    # Store in DataFrame
    if fid_counts is None:
        fid_counts = count_final
    else:
        fid_counts = fid_counts.merge(count_final[['flight_path', 'camera_direction', f'fid_{i}']],
                                      on=['flight_path', 'camera_direction'],
                                      how='outer')

# Ensure correct data types
numerical_cols = ['total_images', 'fid_5', 'fid_6', 'fid_7', 'fid_8']
fid_counts[numerical_cols] = fid_counts[numerical_cols].astype(int)

# Rename fid columns
fid_counts = fid_counts.rename(columns={
    'fid_7': 'fid_N',
    'fid_5': 'fid_W',
    'fid_6': 'fid_E',
    'fid_8': 'fid_S'
})

# Reorder columns
column_order = ['flight_path', 'camera_direction', 'total_images', 'fid_N', 'fid_E', 'fid_S', 'fid_W']
fid_counts = fid_counts[column_order]

# Copy updated DataFrame to clipboard
fid_counts.to_clipboard(excel=True)

print(fid_counts)
