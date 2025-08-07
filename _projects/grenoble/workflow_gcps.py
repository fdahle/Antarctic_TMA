project = "casa_grande"
img_type = "aerial_images"

project_path = f"/data/ATM/grenoble_project/{project}/intermediate/{img_type}/{img_type}.psx"
gcp_path = f"/data/ATM/grenoble_project/{project}/input/{img_type}/preprocessed_images/gcp.csv"

import Metashape

# load the project
doc = Metashape.Document(read_only=False)
doc.open(project_path, ignore_lock=True)
chunk = doc.chunks[0]

# load the gcps
import pandas as pd
gcps = pd.read_csv(gcp_path)

gcps.columns = ['GCP', 'filename', 'img_x', 'img_y', 'x_abs', 'y_abs', 'z_abs',
                'accuracy_x', 'accuracy_y', 'accuracy_z']

# remove .tif from filenames
gcps['filename'] = gcps['filename'].str.replace('.tif', '', regex=False)

import src.sfm_agi2.snippets.add_markers as am

am.add_markers(chunk, gcps, 4326,
               reset_markers=True,
               accuracy_dict=None, direct_projection=True)
doc.save()

# set project to absolute
chunk.crs = Metashape.CoordinateSystem(f"EPSG::{32612}")  # noqa
chunk.camera_crs = Metashape.CoordinateSystem(f"EPSG::{32612}")  # noqa
doc.save()

import src.sfm_agi2.snippets.filter_markers as fm
fm_settings = {
    "marker_lbl": "cg",
    "min_markers": 5,
    "max_error_px": 2,
    "max_error_m": 1
}
#fm.filter_markers(chunk, **fm_settings)
chunk.resetRegion()
doc.save()

chunk.buildDepthMaps()

doc.save()

chunk.buildModel()
doc.save()