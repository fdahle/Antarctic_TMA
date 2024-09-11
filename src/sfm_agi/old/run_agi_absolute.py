# Library imports
import geopandas as gpd
import os
import Metashape
import numpy as np
import pandas as pd
import shutil
from shapely.geometry import Point

# variable imports
import src.base.load_credentials as lc
import src.load.load_image as li
import src.sfm_agi.snippets.export_footprints as cf

# ignore some warnings
os.environ['KMP_WARNINGS'] = '0'

# Constants
PATH_PROJECT_FOLDERS = "/data/ATM/data_1/sfm/agi_projects"
RESUME = False
OVERWRITE = True
DEBUG_PRINT = False

# Variables
resolution_absolute = 2

STEPS = {
    "set_focal_length": True,
    "load_masks": True,
    "match_photos": False,
    "pre_align_cameras": False,
    "import_alignment": True,
    "load_gcps": True,
    "align_cameras": False,
    "build_depth_maps": False,
    "build_dense_cloud": False,
    "build_mesh": False,
    "create_bounding_box": False,
    "build_dem": False,
    "build_orthomosaic": False,
    "build_pointcloud": False
}

DEBUG = {
    "save_positions": False,
    "save_footprints": False
}


def run_agi_absolute(project_name, images, focal_lengths=None,
                     epsg_code=3031):
    if RESUME and OVERWRITE:
        raise ValueError("Both RESUME and OVERWRITE cannot be set to True.")

    # define path to the project folder and the project files
    project_fld = os.path.join(PATH_PROJECT_FOLDERS, project_name)
    project_file_path_relative = project_fld + "/" + project_name + "_relative.psx"
    project_file_path_absolute = project_fld + "/" + project_name + "_absolute.psx"

    # check if the project folder exists
    if not os.path.exists(project_fld):
        raise FileNotFoundError(f"The folder {project_fld} does not exist.")

    # check if the relative project file exists
    if not os.path.exists(project_file_path_relative):
        raise FileNotFoundError(f"The file {project_file_path_relative} does not exist.")

    print("STEPS:")
    print(STEPS)

    print("images:")
    print(images)

    # get the license key
    licence_key = lc.load_credentials("agisoft")['licence']

    # Activate the license
    Metashape.License().activate(licence_key)

    # enable use of gpu
    Metashape.app.gpu_mask = 1
    Metashape.app.cpu_enable = False

    if focal_lengths is None:
        focal_lengths = {}

    # remove the project file if OVERWRITE is set to True
    if OVERWRITE and os.path.exists(project_file_path_absolute):
        print(f"Remove absolute project '{project_name}'")
        # remove the project file
        os.remove(project_file_path_absolute)
        # remove the project files
        shutil.rmtree(project_file_path_absolute.replace(".psx", ".files"))

    # create a metashape project object
    doc = Metashape.Document(read_only=False)  # noqa

    # check if the project already exists
    if os.path.exists(project_file_path_absolute):

        if RESUME is False:
            raise FileExistsError("The project already exists. Set RESUME to True to resume the project.")

        # load the project
        doc.open(project_file_path_absolute, ignore_lock=True)

    else:
        # save the project with file path so that later steps can be resumed
        doc.save(project_file_path_absolute)

    # init output folder
    output_fld = os.path.join(project_fld, "output_absolute")
    if os.path.isdir(output_fld) is False:
        os.makedirs(output_fld)

    # init data folder
    data_fld = os.path.join(project_fld, "data")
    if os.path.isdir(data_fld) is False:
        os.makedirs(data_fld)

    # add a chunk
    if len(doc.chunks) == 0:
        chunk = doc.addChunk()

        # add the images to the chunk
        chunk.addPhotos(images)

    else:
        chunk = doc.chunks[0]

    if STEPS["set_focal_length"]:

        print("Set Focal length")

        # set focal length if given
        for camera in chunk.cameras:
            if camera.label in focal_lengths:
                focal_length = focal_lengths[camera.label]
                camera.sensor.focal_length = focal_length
            else:
                print(f"WARNING: Focal length not given for {camera.label}")
        print("Set Focal length - finished")

    if STEPS["load_masks"]:
        mask_folder = os.path.join(data_fld, "masks_adapted")

        for camera in chunk.cameras:
            mask_path = os.path.join(mask_folder, camera.label + "_mask_adapted.tif")
            if os.path.exists(mask_path):

                # load mask
                mask = li.load_image(mask_path)
                mask = mask.astype(np.uint8)

                # assure that the mask is binary (0 or 255)
                if np.amax(mask) == 1:
                    mask *= 255

                meta_img = Metashape.Image.fromstring(mask, mask.shape[1], mask.shape[0],
                                                      channels=' ', datatype='U8')
                meta_mask = Metashape.Mask()
                meta_mask.setImage(meta_img)
                camera.mask = meta_mask
            else:
                raise FileNotFoundError(f"Mask not found for {camera.label}")

    if STEPS["match_photos"]:

        print("Match photos")

        # create pairs from the cameras
        pairs = []
        num_cameras = len(chunk.cameras)
        for i in range(num_cameras - 1):
            pairs.append((i, i + 1))

        # match photos
        chunk.matchPhotos(generic_preselection=True, reference_preselection=True,
                          keep_keypoints=True,
                          pairs=pairs,
                          filter_mask=True, mask_tiepoints=True,
                          filter_stationary_points=True, reset_matches=True)
        doc.save()

        print("Match photos - finished")

    # align cameras
    if STEPS["pre_align_cameras"]:
        print("Pre-align cameras")

        # align cameras
        chunk.alignCameras(adaptive_fitting=False)
        doc.save()

        print("Pre-align cameras - finished")

    if STEPS["import_alignment"]:
        pc_path = os.path.join(data_fld, "point_cloud.ply")
        chunk.importPointCloud(pc_path, format=Metashape.PointCloudFormatPLY)

        # Import camera calibration and orientation
        camera_path = os.path.join(data_fld, "cameras.txt")
        with open(camera_path, 'r') as f:
            for line in f:
                data = line.strip().split(',')
                label = data[0]
                transform_data = list(map(float, data[1:]))
                transform_data = np.array(transform_data).reshape(4, 4)
                transform_matrix = Metashape.Matrix(transform_data)  # noqa

                # Find corresponding camera in target chunk
                for camera in chunk.cameras:
                    if camera.label == label:
                        camera.transform = transform_matrix
                        break
        doc.save()
    # set the chunk coordinate system
    chunk.crs = Metashape.CoordinateSystem(f"EPSG::{epsg_code}")  # noqa
    doc.save()

    if STEPS["load_gcps"]:

        # load gcps from file
        gcp_path = os.path.join(data_fld, "gcps.csv")
        gcps = pd.read_csv(gcp_path, sep=';')

        # set crs of markers
        chunk.marker_crs = Metashape.CoordinateSystem(f"EPSG::{epsg_code}")  # noqa
        chunk.marker_location_accuracy = [25, 25, 100]

        # multiply the number of gcps by the number of cameras to get the total number of possible markers
        ttl = gcps.shape[0] * len(chunk.cameras)

        # iterate over the gcp dataframe
        for _, row in gcps.iterrows():

            if DEBUG_PRINT:
                print("load", row['GCP'])

            # remove the marker if it is already existing
            mrk = [marker for marker in chunk.markers if marker.label == row['GCP']]
            if len(mrk) == 1:
                chunk.markers.remove(mrk[0])

            # don't add gcps that are likely wrong
            if row['z_abs'] == 0:
                if DEBUG_PRINT:
                    print(" skip - z_abs == 0")
                continue

            # create 3d point
            point_3d = Metashape.Vector([row['x_rel'], row['y_rel'], row['z_rel']])  # noqa

            # transform the point to local coordinates
            point_local = chunk.transform.matrix.inv().mulp(point_3d)

            marker = None
            for camera in chunk.cameras:
                if DEBUG_PRINT:
                    print(" camera", camera.label)
                projection = camera.project(point_local)
                if projection is None:
                    if DEBUG_PRINT:
                        print("  skip - projection is None")
                    continue
                x, y = projection

                if ((0 <= x <= camera.image().width) and
                        (0 <= y <= camera.image().height)):

                    # marker must be created only once
                    if marker is None:
                        marker = chunk.addMarker()
                        marker.label = row['GCP']

                        # set the reference location for the marker
                        marker.reference.location = Metashape.Vector([row['x_abs'], row['y_abs'], row['z_abs']])  # noqa

                    print("  marker", marker.label, "camera", camera.label,
                          "x_abs", row['x_abs'], "y_abs", row['y_abs'], "z_abs", row['z_abs'],
                          "x", x, "y", y, "z", row['z_rel'])

                    # set relative projection for the marker
                    m_proj = Metashape.Marker.Projection(Metashape.Vector([x, y]), True)  # noqa
                    marker.projections[camera] = m_proj  # noqa
                else:
                    if DEBUG_PRINT:
                        print("  skip - projection out of bounds")

        # "https://www.agisoft.com/forum/index.php?topic=7446.0"
        # "https://www.agisoft.com/forum/index.php?topic=10855.0"

        doc.save()

    # align cameras
    if STEPS["align_cameras"]:
        print("Align cameras")

        # align cameras
        chunk.alignCameras(reset_alignment=True, adaptive_fitting=False)
        doc.save()

        print("Align cameras - finished")

    if DEBUG["save_positions"]:
        save_fld = os.path.join(data_fld, "shapes")
        if not os.path.exists(save_fld):
            os.makedirs(save_fld)

        save_path = os.path.join(save_fld, "cameras.shp")

        # prepare geo-dataframe
        data = {
            'camera_label': [],
            'geometry': [],
            'altitude': []
        }

        for camera in chunk.cameras:
            if camera.reference.location:
                # Create a Point geometry for GeoPandas
                x, y, z = camera.reference.location
                data['camera_label'].append(camera.label)
                data['geometry'].append(Point(x, y))
                data['altitude'].append(z)

        gdf = gpd.GeoDataFrame(data, crs=f'EPSG:{epsg_code}')
        gdf.to_file(save_path)

    # build depth maps
    if STEPS["build_depth_maps"]:
        print("Build depth maps")

        chunk.buildDepthMaps()
        doc.save()

        print("Build depth maps - finished")

    # build dense cloud
    if STEPS["build_dense_cloud"]:
        print("Build dense cloud")

        chunk.buildPointCloud()
        doc.save()

        print("Build dense cloud - finished")

    # build mesh
    if STEPS["build_mesh"]:
        print("Build mesh")

        # build mesh
        chunk.buildModel(surface_type=Metashape.Arbitrary)
        doc.save()

        # define output path for the model
        output_model_path = os.path.join(output_fld, project_name + "_model_absolute.obj")

        # define export parameters
        export_params = {
            'path': output_model_path,
        }

        # export the model
        chunk.exportModel(**export_params)

        print("Build mesh - finished")

    if STEPS["create_bounding_box"]:

        center = chunk.region.center
        size = chunk.region.size

        # Calculate the minimum and maximum corners of the bounding box
        min_corner = Metashape.Vector([center.x - size.x / 2,  # noqa
                                       center.y - size.y / 2,
                                       center.z - size.z / 2])
        max_corner = Metashape.Vector([center.x + size.x / 2,  # noqa
                                       center.y + size.y / 2,
                                       center.z + size.z / 2])

        min_corner_abs = chunk.crs.project(chunk.transform.matrix.mulp(min_corner))
        max_corner_abs = chunk.crs.project(chunk.transform.matrix.mulp(max_corner))

        # create 2d vectors
        min_corner_2d = Metashape.Vector([min_corner_abs.x, min_corner_abs.y])  # noqa
        max_corner_2d = Metashape.Vector([max_corner_abs.x, max_corner_abs.y])  # noqa

        # Create the bounding box
        bounding_box = Metashape.BBox(min_corner_2d, max_corner_2d)  # noqa
    else:
        bounding_box = None

    # build DEM
    if STEPS["build_dem"]:

        print("Build DEM")

        # define projection
        projection = Metashape.OrthoProjection()
        projection.crs = chunk.crs

        # set build parameters for the DEM
        build_params = {
            'source_data': Metashape.DataSource.PointCloudData,
            'interpolation': Metashape.Interpolation.EnabledInterpolation,
            'projection': projection,
            'resolution': resolution_absolute
        }

        # add region to build parameters
        if bounding_box is not None:
            build_params['region'] = bounding_box

        # build the DEM
        chunk.buildDem(**build_params)
        doc.save()

        # define output path for the DEM
        output_dem_path = os.path.join(output_fld, project_name + "_dem_absolute.tif")

        # set export parameters for the DEM
        export_params = {
            'path': output_dem_path,
            'source_data': Metashape.ElevationData,
            'image_format': Metashape.ImageFormatTIFF,
            'raster_transform': Metashape.RasterTransformNone,
            'nodata_value': -9999,
            'resolution': resolution_absolute
        }
        if bounding_box is not None:
            export_params['region'] = bounding_box

        chunk.exportRaster(**export_params)

        print("Build DEM - finished")

    # build ortho-mosaic
    if STEPS["build_orthomosaic"]:

        print("Build orthomosaic")

        projection = Metashape.OrthoProjection()
        projection.crs = chunk.crs

        build_params = {
            'surface_data': Metashape.ModelData,
            'blending_mode': Metashape.MosaicBlending,
            'projection': projection,
            'resolution': resolution_absolute
        }
        if bounding_box is not None:
            build_params['region'] = bounding_box

        chunk.buildOrthomosaic(**build_params)
        doc.save()

        output_ortho_path = os.path.join(output_fld, project_name + "_ortho_absolute.tif")

        export_params = {
            'path': output_ortho_path,
            'source_data': Metashape.OrthomosaicData,
            'image_format': Metashape.ImageFormatTIFF,
            'raster_transform': Metashape.RasterTransformNone,
            'nodata_value': -9999,
            'resolution': resolution_absolute
        }
        if bounding_box is not None:
            export_params['region'] = bounding_box

        chunk.exportRaster(**export_params)

        print("Build orthomosaic - finished")

    if STEPS["build_pointcloud"]:
        chunk.buildPointCloud()
        doc.save()

        output_pc_path = os.path.join(output_fld, project_name + "_pointcloud.laz")

        export_params = {
            'path': output_pc_path,
        }
        chunk.exportPointCloud(**export_params)

    if DEBUG["save_footprints"]:

        save_fld = os.path.join(data_fld, "shapes")
        if not os.path.exists(save_fld):
            os.makedirs(save_fld)

        cf.export_footprints(chunk, save_fld)
