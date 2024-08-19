import geopandas as gpd
import simplekml
import os

# Load the shapefile
shapefile_path = '/data/ATM/colombia/data/grids/grid_500.shp'
gdf = gpd.read_file(shapefile_path)

# Reproject to WGS 84 (EPSG:4326)
gdf = gdf.to_crs(epsg=4326)

# Output directory for KML files
output_dir = "/data/ATM/colombia/routes/kml_500"
os.makedirs(output_dir, exist_ok=True)

# Loop through each polygon and create a KML file
for idx, row in gdf.iterrows():
    kml = simplekml.Kml()

    # Check if the geometry is a Polygon or MultiPolygon
    if row.geometry.type == 'Polygon':
        coords = list(row.geometry.exterior.coords)
        kml.newpolygon(name=f"Polygon {idx + 1}", outerboundaryis=coords)
    elif row.geometry.type == 'MultiPolygon':
        for poly in row.geometry:
            coords = list(poly.exterior.coords)
            kml.newpolygon(name=f"Polygon {idx + 1}", outerboundaryis=coords)

    # Save the KML file
    kml_file_path = os.path.join(output_dir, f"polygon_{idx + 1}.kml")
    print(kml_file_path)
    kml.save(kml_file_path)

print(f"Saved {len(gdf)} KML files to '{output_dir}'")
