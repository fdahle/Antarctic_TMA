import pandas as pd
from papers.fid_mark_paper.estimate_altitude import flight_paths
import src.base.connect_to_database as ctd

# Establish database connection
conn = ctd.establish_connection()

# Define the mode and flight paths
mode = "flight_paths"  # can be 'all', 'flight_paths' or 'area'

if mode == "flight_paths":
    flight_paths_10 = ["1821", "1816", "1833", "2137", "1825", "2136",
                       "2143", "1826", "1813", "2141"]
    flight_paths_20 = ["2140", "2073", "1822", "1827", "1684", "2142",
                       "1824", "1846", "2139", "2075"]

    flight_paths = flight_paths_10 + flight_paths_20

    import src.other.extract.extract_ids_by_flightpath as eibf
    image_ids = eibf.extract_ids_by_area(flight_paths, conn=conn)

# Get extracted data for images
sql_string = f"SELECT * FROM images_extracted"
if image_ids is not None:
    sql_string += f" WHERE image_id IN {tuple(image_ids)}"
data_extracted = ctd.execute_sql(sql_string, conn)

# Get fid-mark data for images
sql_string = f"SELECT * FROM images_fid_points"
if image_ids is not None:
    sql_string += f" WHERE image_id IN {tuple(image_ids)}"
data_fid = ctd.execute_sql(sql_string, conn)

# Merge data and create combined altitude/height column
data = data_extracted.merge(data_fid, on='image_id')
data['altitude_height_combined'] = data['altimeter_value'].combine_first(data['height'])

# Define camera directions
camera_directions = ["31L", "32V", "33R"]

# Prepare statistics storage
stats_table = []

# Loop through each flight path and camera direction to calculate statistics
for flight_path in flight_paths:
    data_fp = data[data['image_id'].str.contains(f'CA{flight_path}')]

    if data_fp.empty:
        continue

    total_images = data_fp.shape[0]

    extracted_values = []
    estimated_values = []
    missing_values = []

    for camera in camera_directions:
        data_camera = data_fp[data_fp['image_id'].str.contains(camera)]

        extracted_count = data_camera['altitude_height_combined'].dropna().shape[0]
        estimated_count = data_camera['altimeter_estimated'].sum() + data_camera['height_estimated'].sum()
        missing_count = data_camera.shape[0] - extracted_count

        extracted_values.append(str(extracted_count))
        estimated_values.append(str(estimated_count))
        missing_values.append(str(missing_count))

    # Append data in the required format for LaTeX
    stats_table.append({
        "Flight Path": flight_path,
        "Total Images": total_images,
        "Extracted": "/".join(extracted_values),
        "Estimated": "/".join(estimated_values),
        "Missing": "/".join(missing_values)
    })

# Convert stats to a DataFrame
stats_df = pd.DataFrame(stats_table)

stats_df = stats_df.sort_values(by="Flight Path")

# Generate LaTeX table
latex_table = stats_df.to_latex(
    index=False,
    caption="Flight Path Statistics with Camera Directions",
    label="tab:flight_path_camera_stats",
    column_format="|l|r|c|c|c|",
    header=["Flight Path", "Total Images", "Extracted (31L/32V/33R)", "Estimated (31L/32V/33R)", "Missing (31L/32V/33R)"]
)

print("\nGenerated LaTeX Table:")
print(latex_table)
