import pandas as pd
import numpy as np

import base.load_shape_data as lsd
import base.connect_to_db as ctd

import misc.convert_to_latex as ctl

shape_path = "/data_1/ATM/data_1/papers/paper_georef/tma_points_aoi_small.shp"
data = lsd.load_shape_data(shape_path)

# Get counts of each TMA_num
value_counts = data['TMA_num'].value_counts()

# Filter TMA_NUM that appear 10 or more times
tma_to_keep = value_counts[value_counts >= 10].index

# Keep rows where TMA_NUM is in tma_to_keep
data = data[data['TMA_num'].isin(tma_to_keep)]

# Creating a new column 'image_id' based on 'ENTITY_ID'
data['image_id'] = data['ENTITY_ID'].apply(lambda x: x[0:6] + "32V" + x[-4:])

# get the geo-referenced images
sql_string = "SELECT image_id, footprint_exact, position_error_vector, complexity FROM images_extracted"
data_ex = ctd.get_data_from_db(sql_string)

# merge data
data = pd.merge(data, data_ex[['image_id', 'position_error_vector', 'complexity']], on='image_id', how='left')

# Create a new column 'georeferenced' with default False
data['georeferenced'] = False

# Iterate over each row in 'data'
for index, row in data.iterrows():
    image_id = row['image_id']
    # Check if the image_id is in 'data_ex' and footprint_exact is not None
    footprint = data_ex[data_ex['image_id'] == image_id]['footprint_exact']
    if not footprint.empty and footprint.iloc[0] is not None:
        data.at[index, 'georeferenced'] = True

# calculate error vector length
# Split the 'x;y' string into separate 'x' and 'y' columns
data[['error_x', 'error_y']] = data['position_error_vector'].str.split(';', expand=True)

# Convert 'x' and 'y' columns to numeric data types
data[['error_x', 'error_y']] = data[['error_x', 'error_y']].apply(pd.to_numeric)

# Calculate the vector length using Euclidean distance formula
data['error_length'] = np.sqrt(data['error_x']**2 + data['error_y']**2)

data = data.sort_index()
data = data[['TMA_num', 'image_id', 'georeferenced', 'complexity', 'error_length']]

grouped = data.groupby('TMA_num').agg(
    count=('TMA_num', 'size'),  # Count of all rows
    count_georef=('georeferenced', 'sum'),  # Count of rows where georeferenced is True
    complexity=('complexity', 'mean'),  # Average complexity for each TMA_num group
    error_length=('error_length', 'mean')
)

# round
grouped['complexity'] = np.round(grouped['complexity'], 2)
grouped['count'] = grouped['count'].astype(int)
grouped['count_georef'] = grouped['count_georef'].astype(int)
grouped['error_length'] = np.round(grouped['error_length'], 2)

print(grouped)

ctl.convert_to_latex(grouped)