
import pandas as pd
import numpy as np

# experiment parameters
site = "iceland"  # can be either iceland or casa_grande
dataset = "aerial"  # can be aerial, pc or mc
image_type = "preprocessed"  # can be preprocessed or raw
absolute_bounds = "auto"  # with auto defined by the input shape file
absolute_bounds_epsg = 4326
use_calib_info = True
use_gcps = False
use_coreg = False
use_multi_temp_ba = False

# Safety parameters
overwrite = True
resume = False
steps = {
    "load_data": True,
    "match_images": True,
    "add_data": True,
    "update_cameras": True,
    "create_products": True,
    "export_products": True
}

# camera parameters
use_film_camera = False
fixed_calibration = False

# matching parameters
min_overlap = 0.2  # minimum overlap to consider images for overlapping
min_tp_conf = 0.9 # minimum confidence for tie points to be considered
check_rotated_tps = True  # check for rotated images at tie point detection
min_tps = 100  # minimum number of tie points to consider a pair of images valid

# output
resolution = 2  # resolution of the output products in meters

# set path for the intermediate and final results
intermediate_fld = f"/data/ATM/grenoble_project/{site}/intermediate/{dataset}_{image_type}"
output_fld = f"/data/ATM/grenoble_project/{site}/output/{dataset}_{image_type}"

if overwrite and resume:
    raise ValueError("Cannot set both overwrite and resume to True. Please set one of them to False.")

# set epsg code based on the site
if site == "casa_grande":
    epsg_code = 32612  # UTM zone 12N
elif site == "iceland":
    epsg_code = 32627  # UTM zone 27N
else:
    raise ValueError(f"Unknown site: {site}. Please set a valid site (casa_grande or iceland).")

#%% define the coordinate systems and projections
import Metashape
import pyproj
crs_utm = Metashape.CoordinateSystem(f"EPSG::{epsg_code}")
proj_utm = Metashape.OrthoProjection(crs_utm)
crs_wgs84 = Metashape.CoordinateSystem("EPSG::4326")
proj_wgs84 = Metashape.OrthoProjection(crs_wgs84)

transformer_back = pyproj.Transformer.from_crs(f"EPSG:{epsg_code}", "EPSG:4326", always_xy=True)
transformer = pyproj.Transformer.from_crs("EPSG:4326", f"EPSG:{epsg_code}", always_xy=True)


#%% create project_name based on settings
project_name = f"fdahle"
suffix = None

if site == "casa_grande":
    project_name += "_CG"
elif site == "iceland":
    project_name += "_IL"
else:
    raise ValueError(f"Unknown site: {site}. Please set a valid site (casa_grande or iceland).")

if dataset == "aerial":
    project_name += "_AI"
elif dataset == "pc":
    project_name += "_PC"
elif dataset == "mc":
    project_name += "_MC"
else :
    raise ValueError(f"Unknown dataset: {dataset}. Please set a valid dataset (aerial, pc or mc).")

if image_type == "preprocessed":
    project_name += "_PP"
elif image_type == "raw":
    project_name += "_RA"
else:
    raise ValueError(f"Unknown image type: {image_type}. Please set a valid image type (preprocessed or raw).")

if use_calib_info:
    project_name += "_CY"
else:
    project_name += "_CN"

if use_gcps:
    project_name += "_GY"
else:
    project_name += "_GN"

if use_coreg:
    project_name += "_PY"
else:
    project_name += "_PN"

if use_multi_temp_ba:
    project_name += "_MY"
else:
    project_name += "_MN"

if suffix is not None:
    project_name += f"_{suffix}"

#%% create the project paths
project_psx_path = intermediate_fld + "/" + project_name + ".psx"
project_files_path = intermediate_fld + "/" + project_name + ".files"

import os
import shutil

if os.path.exists(project_psx_path):
    if overwrite:
        os.remove(project_psx_path)
        shutil.rmtree(project_files_path)
        # we then need to set all steps to True
        for key in steps.keys():
            steps[key] = True
    elif resume:
        # do nothing
        pass
    else:
        raise FileExistsError(f"Project file already exists: {project_psx_path}")
else:
    # set all steps to True if the project file does not exist
    for key in steps.keys():
        steps[key] = True

footprint_path = f"/data/ATM/grenoble_project/{site}/input/{dataset}/images_footprint.shp"
import src.load.load_shape_data as lsd
footprints = lsd.load_shape_data(footprint_path)
if absolute_bounds == "auto":
    absolute_bounds = footprints.total_bounds
if absolute_bounds_epsg == "auto":
    # TODO FIX THIS
    absolute_bounds_epsg = footprints.crs.to_epsg()

#%% """ Loading the data """
if steps["load_data"]:
    import glob

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

        mask = None
        mask_rotated = None

        pass # TODO OWN MASK

    # load the footprints and get a list of geometries
    footprints = lsd.load_shape_data(footprint_path)
    if isinstance(absolute_bounds, str) and absolute_bounds == "auto":
        absolute_bounds = footprints.total_bounds

    # second column is always the ID column (first is geometry)
    id_col = footprints.columns[1]

    geom_map = {row[id_col].replace('.tif', ''): row['geometry'] for _, row in footprints.iterrows()}
    polygons = [geom_map[img] for img in lst_image_names if img in geom_map]
else:
    # init all variables as None so that the ide does not complain about undefined variables
    img_folder, lst_image_paths, lst_image_names = None, None, None
    mask_path, mask, mask_rotated = None, None, None
    polygons = None

#%% Create the project

# create a new Metashape project or open an existing one
doc = Metashape.Document(read_only=False)
if os.path.exists(project_psx_path):
    doc.open(project_psx_path)
    chunk = doc.chunks[0]
else:
    doc.save(project_psx_path)
    chunk = doc.addChunk()

    # add the images to the chunk
    chunk.addPhotos(lst_image_paths, strip_extensions=True)

#%% Match the images
if steps["match_images"]:
    # find tie-points between the images
    import numpy as np
    from tqdm import tqdm

    import src.base.find_overlapping_images_geom as foig
    import src.base.find_tie_points as ftp
    import src.base.rotate_points as rp

    # define saving folder for tie points
    tp_folder = os.path.join(intermediate_fld, "tie_points")
    os.makedirs(tp_folder, exist_ok=True)

    # find the overlapping images
    overlapping_images = foig.find_overlapping_images_geom(lst_image_names, polygons,
                                                           min_overlap=min_overlap)

    # init the tie point detector
    tpd = ftp.TiePointDetector("lightglue", min_conf=min_tp_conf,
                               catch=False, verbose=True, display=True)

    # init progress bar
    num_pairs = sum(len(lst) for lst in overlapping_images.values())

    # flag to check if tie points were updated
    tps_updated = False

    # track which tps are used
    tp_file_list = []

    # iterate over the overlapping images
    print("Look for tie-points: ")
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

    # Create bundler
    import src.sfm_agi2.snippets.create_bundler as cb
    import warnings

    # define path to the bundler folder
    bundler_folder = os.path.join(intermediate_fld, "bundler")
    os.makedirs(bundler_folder, exist_ok=True)
    bundler_pth = os.path.join(bundler_folder, f"bundler_{int(min_overlap*10)}.out")

    if tps_updated and os.path.exists(bundler_pth):
        os.remove(bundler_pth)

    # check if bundler file already exists
    if os.path.isfile(bundler_pth):
        # skip
        print("Bundler file already exists, skipping creation.")
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
                except (Exception,):
                    data = np.empty((0, 5))  # empty array if file is not found
                try:
                    data_r = np.loadtxt(tp_pth_r,  delimiter=',', skiprows=1)
                except (Exception,):
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
        # rename the bundler file to include the overlap
        os.rename(os.path.join(bundler_folder, "bundler.out"), bundler_pth)

    # load the bundler file
    bundler_pth = os.path.join(bundler_folder, f"bundler_{int(min_overlap * 10)}.out")
    chunk.importCameras(bundler_pth, format=Metashape.CamerasFormatBundler)
    doc.save()

#%% add data to the project
if steps["add_data"]:
    # add the extrinsics to the cameras
    if use_calib_info:
        path_extrinsics = f"/data/ATM/grenoble_project/{site}/input/{dataset}/camera_model_extrinsics.csv"

        if os.path.isfile(path_extrinsics) is False:
            print(f"Extrinsics file not found: {path_extrinsics}!")
        else:
            ext = pd.read_csv(path_extrinsics)
            print("Adding extrinsics from file")

            # convert the lon/lat to UTM coordinates
            utm_coords = transformer.transform(ext['lon'].values, ext['lat'].values)
            ext['x_abs'] = utm_coords[0]  # UTM Easting
            ext['y_abs'] = utm_coords[1]  # UTM Northing

            # normalize key to match camera labels (you added photos with strip_extensions=True)
            ext['key'] = ext['image_file_name'].str.replace('.tif', '', regex=False)

            # build quick lookup: { 'ARBCSRD0001' : {'lon':..., 'lat':..., 'alt':...}, ... }  # noqa
            ext_map = ext.set_index('key')[['x_abs', 'y_abs', 'alt']].to_dict('index')

            # set the CRS of the camera
            chunk.camera_crs = Metashape.CoordinateSystem(f"EPSG::{epsg_code}")

            # (optional) give very loose default accuracies so they help initialize but don't dominate BA
            default_xy_acc_m = 100.0
            default_z_acc_m  = 200.0

            missing = []
            for cam in chunk.cameras:
                if cam.label in ext_map:
                    x = float(ext_map[cam.label]['x_abs'])
                    y = float(ext_map[cam.label]['y_abs'])
                    z = float(ext_map[cam.label]['alt'])

                    cam.reference.enabled = True
                    cam.reference.location = Metashape.Vector([x, y, z])
                    cam.reference.accuracy = Metashape.Vector([default_xy_acc_m,
                                                               default_xy_acc_m,
                                                               default_z_acc_m])
                else:
                    missing.append(cam.label)

            if missing:
                print(f"[Extrinsics] No entry for {len(missing)} images, e.g.: {missing[:5]}")

    # create a metashape mask object
    if mask_path is not None:
        mask255 = mask.astype(np.uint8)*255
        mask_image = Metashape.Image.fromstring(mask255, mask.shape[1],  # noqa
                                                  mask.shape[0], channels=' ', datatype='U8')
        mask_obj = Metashape.Mask()
        mask_obj.setImage(mask_image)

        # set mask for every image
        for camera in chunk.cameras:
            camera.mask = mask_obj

    # add existing internal
    fit_focal_length = True  # fit focal length
    if use_calib_info:
        print("Adding intrinsics from file")

        # focal length is provided -> no fitting
        fit_focal_length = False  # noqa

        # set the path to the internal calibration file
        path_intrinsics = f"/data/ATM/grenoble_project/{site}/input/{dataset}/camera_model_intrinsics.csv"

        if not os.path.isfile(path_intrinsics):
            print(f"Intrinsics file not found: {path_intrinsics}!")
        else:
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

            # convert mm to pixel
            mm2px = lambda v: v / intrinsics['pixel_pitch']  # convert mm to pixel

            # iterate all cameras in the chunk
            for camera in chunk.cameras:

                # TODO: SUPPORT DIFFERENT CAMERAS

                # set to film camera
                camera.sensor.film_camera = use_film_camera
                camera.sensor.fixed_calibration = fixed_calibration

                # set some values
                if "focal_length" in intrinsics:
                    camera.sensor.focal_length = intrinsics['focal_length']
                if "pixel_pitch" in intrinsics:
                    camera.sensor.pixel_size = (intrinsics['pixel_pitch'], intrinsics['pixel_pitch'])

                if use_film_camera:
                    camera.sensor.height = 0
                    camera.sensor.width = 0
                else:
                    camera.sensor.height = camera.photo.image().height
                    camera.sensor.width = camera.photo.image().width

                if "pixel_pitch" in intrinsics:
                    if "focal_length" in intrinsics:
                        camera.sensor.calibration.f = mm2px(camera.sensor.focal_length)
                    if "principal_point_x_mm" in intrinsics and "principal_point_y_mm" in intrinsics:
                        camera.sensor.calibration.cx = mm2px(intrinsics['principal_point_x_mm'])
                        camera.sensor.calibration.cy = mm2px(intrinsics['principal_point_y_mm'])

            doc.save()

            # if applicable, set the fiducials
            if fiducials is not None:

                # get a sensor from any camera
                sensor = chunk.sensors[0]
                if len(chunk.sensors) > 1:
                    print("Warning: More than one sensor found")

                fid_map  = { }
                # iterate over all positions
                for pos in ["corner_top_left", "corner_top_right", "corner_bottom_left", "corner_bottom_right"]:

                    f = chunk.addMarker()
                    f.type= Metashape.Marker.Type.Fiducial
                    f.sensor = sensor
                    f.label = f"fid_{pos}"

                    # Set fiducial coordinates in mm (relative to film frame center)
                    f.reference.location = Metashape.Vector([
                        intrinsics[f"{pos}_x_mm"],
                        #intrinsics[f"{pos}_y_mm"],
                        -intrinsics[f"{pos}_y_mm"],
                        1  # third value is usually 1mm, standard for fiducials
                    ])
                    fid_map[pos] = f

                for camera in chunk.cameras:

                    row = fiducials.loc[fiducials['image_id'] == camera.label + ".tif"]

                    if row.empty:
                        print(f"Warning: No fiducial data found for camera {camera.label}. Skipping.")
                        continue

                    for pos, f in fid_map.items():
                        # Set fiducial projection in pixel coordinates
                        f.projections[camera] = Metashape.Marker.Projection(
                            Metashape.Vector([
                                row[f"{pos}_x"].values[0],  # x_pix
                                row[f"{pos}_y"].values[0]   # y_pix
                            ]),
                            True  # 'True' indicates the projection is enabled
                        )
            doc.save()

    # add provided gcps
    import src.sfm_agi2.snippets.add_markers as am

    if use_gcps:
        print("Adding GCPs from file")

        # set path to the GCPs
        path_gcps = (f"/data/ATM/grenoble_project/{site}/input/{dataset}/{image_type}_images/"
                          f"gcp.csv")

        # read the GCPs from the CSV file and set the columns
        gcps = pd.read_csv(path_gcps)
        gcps.columns = ['GCP', 'filename', 'img_x', 'img_y', 'x_abs', 'y_abs', 'z_abs',
                        'accuracy_x', 'accuracy_y', 'accuracy_z']

        utm_coords = transformer.transform(gcps['x_abs'].values, gcps['y_abs'].values)
        gcps['x_abs'] = utm_coords[0]  # UTM Easting
        gcps['y_abs'] = utm_coords[1]  # UTM Northing

        # remove .tif from filenames
        gcps['filename'] = gcps['filename'].str.replace('.tif', '', regex=False)

        am.add_markers(chunk, gcps, epsg_code, reset_markers=False,
                       accuracy_dict=None, direct_projection=True)  # noqa

        doc.save()

#%% update the cameras
if steps["update_cameras"]:
    if use_calib_info is False and use_gcps is False:
        print("Warning: Model will not be georeferenced, ")
    else:
        chunk.crs = Metashape.CoordinateSystem(f"EPSG::{epsg_code}")

    if use_film_camera:
        for camera in chunk.cameras:
            if camera.sensor.film_camera is False:
                raise ValueError(f"Camera {camera.label} is not a film camera")

    # align the cameras
    chunk.alignCameras(reset_alignment=True, adaptive_fitting=True, subdivide_task=False)
    doc.save()

    for camera in chunk.cameras:
        if use_film_camera and camera.sensor.film_camera is False:
            raise ValueError(f"Camera {camera.label} is not a film camera")
        elif not use_film_camera and camera.sensor.film_camera is True:
            raise ValueError(f"Camera {camera.label} is a film camera")

    # optimize the cameras
    chunk.optimizeCameras(fit_f=fit_focal_length)  # noqa
    #chunk.updateTransform()
    doc.save()

#%% now create all the products
if steps["create_products"]:

    # transform absolute bounds to UTM coordinates
    print(absolute_bounds)
    absolute_bounds = np.array(absolute_bounds,  dtype=float)
    minx, miny, maxx, maxy = map(float, absolute_bounds)
    xs = [minx, maxx, maxx, minx]
    ys = [miny, miny, maxy, maxy]
    X, Y = transformer.transform(xs, ys)
    absolute_bounds = [float(np.min(X)), float(np.min(Y)),
                  float(np.max(X)), float(np.max(Y))]
    print(absolute_bounds)
    # create region based on the absolute bounds
    region = Metashape.BBox(min=(Metashape.Vector(absolute_bounds[:2])), max=(Metashape.Vector(absolute_bounds[2:])))

    chunk.buildDepthMaps(filter_mode=Metashape.FilterMode.AggressiveFiltering)
    doc.save()
    chunk.buildModel(source_data=Metashape.DepthMapsData, surface_type=Metashape.SurfaceType.Arbitrary,
                     face_count=Metashape.FaceCount.HighFaceCount, interpolation=Metashape.EnabledInterpolation)
    doc.save()
    chunk.buildPointCloud(source_data=Metashape.DepthMapsData, point_colors=True, point_confidence=True)
    doc.save()
    print(f"Chunk crs: {chunk.crs}")
    print(f"Chunk region size: {chunk.region.size}")
    print(f"Chunk region center: {chunk.region.center}")
    print(f"Chunk region rotation: {chunk.region.rot}")
    print(f"Chunk transform: {chunk.transform.matrix}")
    chunk.buildDem(resolution=resolution, projection=proj_utm, region=region)
    doc.save()
    chunk.buildOrthomosaic(surface_data=Metashape.DataSource.ElevationData,
                           blending_mode=Metashape.BlendingMode.MosaicBlending,
                           resolution=resolution, projection=proj_utm)
    doc.save()

#%% export the products
if steps["export_products"]:

    # define compression
    compression = Metashape.ImageCompression()
    compression.tiff_compression = Metashape.ImageCompression.TiffCompressionLZW
    compression.tiff_big = True

    dem_path = os.path.join(output_fld, f"{project_name}_dem.tif")
    ortho_path = os.path.join(output_fld, f"{project_name}_orthoimage.tif")
    dense_pc_path = os.path.join(output_fld, f"{project_name}_dense_pointcloud.laz")
    sparse_pc_path = os.path.join(output_fld, f"{project_name}_sparse_pointcloud.laz")
    extrinsics_path = os.path.join(output_fld, f"{project_name}_extrinsics.csv")
    intrinsics_path = os.path.join(output_fld, f"{project_name}_intrinsics.csv")

    # export DEM
    chunk.exportRaster(path=dem_path, source_data=Metashape.ElevationData,
                       format=Metashape.RasterFormat.RasterFormatTiles,
                       image_format=Metashape.ImageFormat.ImageFormatTIFF,
                       projection=proj_wgs84, nodata_value=-32767,
                       save_alpha=False, image_compression=compression)

    # export Ortho
    chunk.exportRaster(path=ortho_path, source_data=Metashape.OrthomosaicData,
                       format=Metashape.RasterFormat.RasterFormatTiles,
                       image_format=Metashape.ImageFormat.ImageFormatTIFF,
                       projection=proj_wgs84, nodata_value=-32767,
                       save_alpha=False, image_compression=compression)

    # export dense point cloud
    chunk.exportPointCloud(path=dense_pc_path, source_data=Metashape.PointCloudData,
                           save_point_color=True, save_point_confidence=True,
                           crs=crs_wgs84, format=Metashape.PointCloudFormatLAZ)

    # export sparse point cloud
    chunk.exportPointCloud(sparse_pc_path, source_data=Metashape.TiePointsData,
                           save_point_color=True, save_point_confidence=True,
                           crs=crs_wgs84, format=Metashape.PointCloudFormatLAZ)

    # get camera intrinsics and extrinsics
    intr_rows = []
    extr_rows = []

    T = chunk.transform.matrix

    for camera in chunk.cameras:

        # get name of the image
        img_name = camera.label + ".tif"

        # get intrinsics
        calib = camera.sensor.calibration
        focal_length = calib.f
        cx, cy = calib.cx, calib.cy
        k1, k2, k3 = calib.k1, calib.k2, calib.k3
        p1, p2 = calib.p1, calib.p2
        intr_rows.append([img_name, focal_length, cx, cy, k1, k2, k3, p1, p2])

        # get extrinsics
        x, y, z = chunk.crs.project(T.mulp(camera.center))
        # convert to lon/lat/alt
        lon, lat = transformer_back.transform(x, y)
        extr_rows.append([img_name, lon, lat, z])

    # build dataframes
    intrinsics = pd.DataFrame(intr_rows, columns=['image_file_name', 'focal_length', 'cx', 'cy',
                                                  'k1', 'k2', 'k3', 'p1', 'p2'])
    extrinsics = pd.DataFrame(extr_rows, columns=['image_file_name', 'lon', 'lat', 'alt'])

    # export intrinsics and extrinsics
    print("Export Intrinsics: path = ", intrinsics_path)
    intrinsics.to_csv(intrinsics_path, index=False)
    print("Export Extrinsics: path = ", extrinsics_path)
    extrinsics.to_csv(extrinsics_path, index=False)