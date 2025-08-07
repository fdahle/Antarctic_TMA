# experiment parameters
site = "iceland"  # can be either iceland or casa_grande
dataset = "aerial"  # can be aerial, pc or mc
image_type = "preprocessed"  # can be preprocessed or raw
use_calib_info = True
use_gcps = True
use_coreg = False
use_multi_temp_ba = False

# PARAMS
min_overlap = 0.2  # minimum overlap to consider images for overlapping
min_tp_conf = 0.9 # minimum confidence for tie points to be considered
check_rotated_tps = True  # check for rotated images at tie point detection
min_tps = 100  # minimum number of tie points to consider a pair of images valid

# set path for the intermediate results
intermediate_fld = f"/data/ATM/grenoble_project/{site}/intermediate/{dataset}_{image_type}"

# set epsg code based on the site
if site == "casa_grande":
    epsg_code = 32612  # UTM zone 12N
elif site == "iceland":
    epsg_code = 32627  # UTM zone 27N

# create project_name based on settings
project_name = f"project"
if use_calib_info:
    project_name += "_CY_1"
else:
    project_name += "_CY_0"
if use_gcps:
    project_name += "_GCP_1"
else:
    project_name += "_GCP_0"
if use_coreg:
    project_name += "_COREG_1"
else:
    project_name += "_COREG_0"
if use_multi_temp_ba:
    project_name += "_MTBA_1"
else:
    project_name += "_MTBA_0"

project_psx_path = intermediate_fld + "/" + project_name + ".psx"
project_files_path = intermediate_fld + "/" + project_name + ".files"

import os
import shutil
overwrite=True
if overwrite:
    if os.path.exists(project_psx_path):
        os.remove(project_psx_path)
    if os.path.exists(project_files_path):
        shutil.rmtree(project_files_path)

#%% """ Loading the data """
import glob
import numpy as np

import src.base.rotate_image as ri
import src.load.load_image as li
import src.load.load_shape_data as lsd

# define the correct path for different locations
img_folder = f"/data/ATM/grenoble_project/{site}/input/{dataset}/{image_type}_images/images/"
mask_path = f"/data/ATM/grenoble_project/{site}/input/{dataset}/{image_type}_images/image_mask.tif"
footprint_path = f"/data/ATM/grenoble_project/{site}/input/{dataset}/images_footprint.shp"

# reset mask for raw images
if image_type == "raw" or dataset == "pc" or dataset == "mc":
    mask_path = None

# load the image names
lst_image_paths = glob.glob(img_folder + "*.tif")
lst_image_names = [p.split("/")[-1][:-4] for p in lst_image_paths]
lst_image_names.sort()

# load the mask and save a rotated version as well
if mask_path is not None:
    mask = li.load_image(mask_path)

    # convert mask to 0 and 1
    mask = mask / np.amax(mask)

    # check if we have more filtered pixels than valid pixels (then we must invert)
    num_ones = np.count_nonzero(mask == 1)
    num_zeros = np.count_nonzero(mask == 0)
    if num_zeros > num_ones:
        mask = 1 - mask

    # assure the mask is binary
    mask[mask != np.amax(mask)] = 0  # set all non-maximum values to 0

    # final check that only 0 and 1 are present
    if not np.all(np.isin(mask, [0, 1])):
        raise ValueError("Mask should only contain 0 and 1 values!")

    mask_rotated = ri.rotate_image(mask, 180)
else:
    pass # TODO OWN MASK

# load the footprints and get a list of geometries
footprints = lsd.load_shape_data(footprint_path)
id_col = 'Image ID' if 'Image ID' in footprints.columns else 'Entity  ID'
geom_map = {row[id_col].replace('.tif', ''): row['geometry'] for _, row in footprints.iterrows()}
polygons = [geom_map[img] for img in lst_image_names if img in geom_map]

#%% Find tie points between the images
import numpy as np
from tqdm import tqdm

import src.base.find_overlapping_images_geom as foig
import src.base.find_tie_points as ftp
import src.base.rotate_points as rp

# define saving folder for tie points
tp_folder = os.path.join(intermediate_fld, "tie_points")

# find the overlapping images
overlapping_images = foig.find_overlapping_images_geom(lst_image_names, polygons,
                                                       min_overlap=min_overlap)

# init the tie point detector
tpd = ftp.TiePointDetector("lightglue", min_conf=min_tp_conf,
                           catch=False, verbose=True, display=False)

# init progress bar
num_pairs = sum(len(lst) for lst in overlapping_images.values())

# flag to check if tie points were updated
tps_updated = False

# track which tps are used
tp_file_list = []

# iterate over the overlapping images
print(" Look for tie-points: ")
with tqdm(total=num_pairs) as pbar:
    for id_img_1, other_images in overlapping_images.items():

        # define image path
        img_1_path = img_folder + id_img_1 + ".tif"
        img_1 = None

        for id_img_2 in other_images:
            pbar.update(1)

            # define the output path for the tie points
            output_path = os.path.join(tp_folder, f"{id_img_1}_{id_img_2}.txt")

            # skip already existing tie points
            if os.path.isfile(output_path):
                pbar.set_postfix_str(f"Skipping existing tie points for: {id_img_1}, {id_img_2}")
                tp_file_list.append(os.path.basename(output_path))
                continue

            if img_1 is None:
                # load the first image
                img_1 = li.load_image(img_1_path)

            # load the second image
            img_2_path = img_folder + id_img_2 + ".tif"
            img_2 = li.load_image(img_2_path)

            # find tie points between the two images
            pbar.set_postfix_str(f"Finding tie points for: {id_img_1}, {id_img_2}")
            tps, conf = tpd.find_tie_points(img_1, img_2, mask1=mask, mask2=mask)

            # also check rotated images
            if check_rotated_tps:
                pbar.set_postfix_str(f"Finding rotated tie points for: {id_img_1}, {id_img_2}")
                imf_2_rotated, rot_mat = ri.rotate_image(img_2, 180,
                                                         return_rot_matrix=True)
                tps_rot, conf_rot = tpd.find_tie_points(img_1, imf_2_rotated,
                                                        mask1=mask, mask2=mask_rotated)

                # check if there are more tie points in the rotated version
                if tps_rot.shape[0] > tps.shape[0]:
                    tps = tps_rot
                    conf = conf_rot

                    # rotate the tie points back to the original image
                    tps[:, 2:4] = rp.rotate_points(tps[:, 2:4], rot_mat, invert=True)

            # merge tps and conf
            merged = np.hstack((tps, conf.reshape(-1, 1)))

            # save the tie points
            np.savetxt(output_path, merged, fmt='%.6f', delimiter=',', header='x1,y1,x2,y2,conf', comments='')
            tp_file_list.append(os.path.basename(output_path))
            tps_updated = True

#%% Create bundler
import src.sfm_agi2.snippets.create_bundler as cb
import warnings

# define path to the bundler folder
bundler_folder = os.path.join(intermediate_fld, "bundler")
bundler_pth = os.path.join(bundler_folder, f"bundler_{min_overlap*10}.out")

if tps_updated and os.path.exists(bundler_pth):
    os.remove(bundler_pth)

# check if bundler file already exists
if os.path.isfile(bundler_pth):
    # skip
    pass
else:
    # load all tie points and confidence values
    tp_dict = {}
    conf_dict = {}
    for tp_file in tp_file_list:

        # get the two image names from the tie point file name
        img1, img2 = tp_file.split("_")[:2]
        img2 = img2.replace(".txt", "")

        # create tuple for the image pair
        img_pair = (img1, img2)
        img_pair_r = (img2, img1)

        tp_pth = os.path.join(tp_folder, tp_file)
        tp_pth_r = os.path.join(tp_folder, f"{img2}_{img1}.txt")

        # check if the image pair is already in the dictionary (also reversed)
        if img_pair in tp_dict or img_pair_r in tp_dict:
            continue

        # load the tie points
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            try:
                data = np.loadtxt(tp_pth, delimiter=',', skiprows=1)
            except:
                data = np.empty((0, 5))  # empty array if file is not found
            try:
                data_r = np.loadtxt(tp_pth_r,  delimiter=',', skiprows=1)
            except:
                data_r = np.empty((0, 5)) # empty array if file is not found

        # if the reversed tie points have more points, use them
        if data.shape[0] < data_r.shape[0]:
            img_pair = img_pair_r
            data = data_r

        # skip if there are too few tie points
        if data.shape[0] < min_tps:
            continue

        # first 4 columns are the tie points, last column is the confidence
        tps = data[:, :4]
        conf = data[:, 4]

        tp_dict[img_pair] = tps
        conf_dict[img_pair] = conf

    cb.create_bundler(img_folder, bundler_folder, tp_dict, conf_dict,)

#%% Create the project
import Metashape
import pandas as pd

# create a new Metashape project
doc = Metashape.Document(read_only=False)
doc.save(project_psx_path)
chunk = doc.addChunk()

# add the images to the chunk
chunk.addPhotos(lst_image_paths, strip_extensions=True)

# create a metashape mask object
if mask_path is not None:
    mask255 = mask.astype(np.uint8)*255
    mask_image = Metashape.Image.fromstring(mask255, mask.shape[1],  # noqa
                                              mask.shape[0], channels=' ', datatype='U8')
    mask_obj = Metashape.Mask()
    mask_obj.setImage(mask_image)

    # set mask for every image
    for camera in chunk.cameras:
        camera_mask = mask_obj

#%% add existing internal
if use_calib_info:

    # set the path to the internal calibration file
    path_intrinsics = f"/data/ATM/grenoble_project/{site}/input/{dataset}/camera_model_intrinsics.csv"

    import pandas as pd
    intrinsics = pd.read_csv(path_intrinsics)

    # get first row of the intrinsics as a dict
    intrinsics = intrinsics.iloc[0].to_dict()

    # for aerial images we have fiducials as well
    if dataset == 'aerial':
        path_fiducials = (f"/data/ATM/grenoble_project/{site}/input/{dataset}/{image_type}_images/"
                          f"detected_fiducial_markers.csv")
        fiducials = pd.read_csv(path_fiducials)
    else:
        fiducials = None

    # iterate all cameras in the chunk
    for camera in chunk.cameras:

        # set to film camera
        camera.sensor.film_camera = True
        camera.sensor.fixed_calibration = False

        # set some values
        camera.sensor.focal_length = intrinsics['focal_length']
        camera.sensor.pixel_size = (intrinsics['pixel_pitch'], intrinsics['pixel_pitch'])
        camera.sensor.height = camera.photo.image().height
        camera.sensor.width = camera.photo.image().width

        camera.sensor.calibration.cx = intrinsics['principal_point_x_mm']
        camera.sensor.calibration.cy = intrinsics['principal_point_y_mm']

        # if applicable, set the fiducials
        if fiducials is not None:

            # TODO : PRINCIPAL POINT

            # iterate over all positions
            for pos in ["corner_top_left", "corner_top_right", "corner_bottom_left", "corner_bottom_right"]:

                f = chunk.addMarker()
                f.type= Metashape.Marker.Type.Fiducial
                f.sensor = camera.sensor
                f.label = camera.label + "_" + pos

                # Set fiducial coordinates in mm (relative to film frame center)
                f.reference.location = Metashape.Vector([
                    intrinsics[f"{pos}_x_mm"],
                    -intrinsics[f"{pos}_y_mm"],
                    1  # third value is usually 1mm, standard for fiducials
                ])

                # Set fiducial projection in pixel coordinates
                f.projections[camera] = Metashape.Marker.Projection(
                    Metashape.Vector([
                        fiducials.loc[fiducials['image_id'] == camera.label + ".tif", f"{pos}_x"].values[0],  # x_pix
                        fiducials.loc[fiducials['image_id'] == camera.label + ".tif", f"{pos}_y"].values[0]   # y_pix
                    ]),
                    True  # 'True' indicates the projection is enabled
                )
    doc.save()

#%% load the bundler file
bundler_pth = os.path.join(bundler_folder, f"bundler_{min_overlap*10}.out")
chunk.importCameras(bundler_pth, format=Metashape.CamerasFormatBundler)
doc.save()

#%% add provided gcps
import src.sfm_agi2.snippets.add_markers as am

if use_gcps:

    # set path to the GCPs
    path_gcps = (f"/data/ATM/grenoble_project/{site}/input/{dataset}/{image_type}_images/"
                      f"gcp.csv")

    # read the GCPs from the CSV file and set the columns
    gcps = pd.read_csv(path_gcps)
    gcps.columns = ['GCP', 'filename', 'img_x', 'img_y', 'x_abs', 'y_abs', 'z_abs',
                    'accuracy_x', 'accuracy_y', 'accuracy_z']

    # remove .tif from filenames
    gcps['filename'] = gcps['filename'].str.replace('.tif', '', regex=False)

    am.add_markers(chunk, gcps, 4326, reset_markers=True,
                   accuracy_dict=None, direct_projection=True)  # noqa

    # set project to absolute
    chunk.crs = Metashape.CoordinateSystem(f"EPSG::{epsg_code}")  # noqa
    chunk.camera_crs = Metashape.CoordinateSystem(f"EPSG::{epsg_code}")  # noqa

    doc.save()

#%% update the cameras

# align the cameras
chunk.alignCameras(reset_alignment=True, adaptive_fitting=True, subdivide_task=False)
doc.save()

# optimize the cameras
chunk.optimizeCameras()
chunk.updateTransform()
doc.save()

#%% now create all the products
chunk.buildDepthMaps(filter_mode=Metashape.FilterMode.AggressiveFiltering)
doc.save()
chunk.buildModel(source_data=Metashape.DepthMapsData, surface_type=Metashape.SurfaceType.Arbitrary,
                 face_count=Metashape.FaceCount.HighFaceCount, interpolation=Metashape.EnabledInterpolation)
doc.save()
chunk.buildPointCloud(source_data=Metashape.DepthMapsData, point_colors=True, point_confidence=True)
doc.save()
chunk.buildDem()
doc.save()
chunk.buildOrthomosaic(surface_data=Metashape.DataSource.ModelData, blending_mode=Metashape.BlendingMode.MosaicBlending)
doc.save()

#%% export the products
