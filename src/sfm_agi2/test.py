import Metashape

project_psx_path = "/data/ATM/data_1/sfm/agi_projects2/test/test.psx"

doc = Metashape.Document()
doc.open(project_psx_path, ignore_lock=True)
chunk = doc.chunk

# Function to compute and print the extent
def print_extent(raster, label):
    if raster is None:
        print(f"No {label} available in the chunk.")
        return

    print(raster.left, raster.top, raster.right, raster.bottom)


# Get Orthomosaic extent
print_extent(chunk.orthomosaic, "Orthomosaic")
print_extent(chunk.elevation, "Elevation")