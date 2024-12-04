import geopandas as gpd
import pandas as pd
from shapely.validation import make_valid

import os
os.environ['PROJ_LIB'] = '/home/fdahle/miniconda3/envs/tma_env/share/proj/'


# Read the GLIMS dataset
glims = gpd.read_file('/home/fdahle/Desktop/glims/glaciers_ap_names.shp')
print(glims.shape)

# Validate and fix geometries
def validate_geometry(geom):
    if not geom.is_valid:
        # Attempt to fix invalid geometry
        try:
            return geom.buffer(0)  # Clean geometry
        except Exception:
            return make_valid(geom)  # Use Shapely's make_valid for complex fixes
    return geom

# Apply validation to all geometries
glims['geometry'] = glims['geometry'].apply(validate_geometry)

# Ensure 'anlys_time' is in datetime format
glims['anlys_time_f'] = pd.to_datetime(glims['anlys_time'])

# Sort by 'anlys_time' to ensure attributes from the newest row are retained after merging
glims = glims.sort_values(by='anlys_time_f', ascending=False)

# Dissolve geometries by 'glac_name' and 'line_type', retaining the newest attributes
merged_glims = glims.dissolve(
    by=['glac_name', 'line_type'],
    as_index=False,
    aggfunc='first'  # Use attributes from the first (newest) row in each group
)

# Ensure the output GeoDataFrame has a CRS defined
merged_glims = merged_glims.set_crs(glims.crs, allow_override=True)

# drop the 'anlys_time_f' column
merged_glims = merged_glims.drop(columns=['anlys_time_f'])
print(merged_glims.shape)

# save the filtered glims
merged_glims.to_file('/home/fdahle/Desktop/glims/glaciers_ap_names_filtered.shp')

# get entry where line_type is 'glac_bound'
glac_bound = merged_glims[merged_glims['line_type'] == 'glac_bound']

# save the glacier boundaries
glac_bound.to_file('/home/fdahle/Desktop/glims/final_glaciers.shp')

"""
# print unique and counts by 'glac_name'
val_counts = glims['glac_name'].value_counts()

small_glac_name = val_counts[val_counts == 4]
print(small_glac_name)

# select a glacier with 4 polygons
filtered_glims = glims[glims['glac_name'] == small_glac_name.index[0]]

# Check if all rows are identical
if not filtered_glims.eq(filtered_glims.iloc[0]).all(axis=0).all():
    # Find differing columns
    differing_cols = filtered_glims.nunique() > 1  # Columns with more than 1 unique value
    differing_cols = differing_cols[differing_cols].index  # Get column names

    # Display the differences
    differences = filtered_glims[differing_cols]
    print("Differing columns and values:")
    print(differences)
else:
    print("All rows are identical.")

# get geometry of glacier as

"""