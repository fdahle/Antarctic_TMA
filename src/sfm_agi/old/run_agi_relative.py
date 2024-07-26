# Library imports
import os
import Metashape
import numpy as np
import pandas as pd
import shutil
import sys
import json
from tqdm import tqdm

# Local imports
import src.base.load_credentials as lc
import src.load.load_image as li
import src.export.export_thumbnail as et
import src.sfm_agi.snippets.create_adapted_mask as cam
import src.sfm_agi.snippets.export_gcps as eg
import src.sfm_agi.snippets.save_key_points as skp
import src.sfm_agi.snippets.save_tie_points as stp

# ignore some warnings
os.environ['KMP_WARNINGS'] = '0'

# Constants
PATH_PROJECT_FOLDERS = "/data/ATM/data_1/sfm/agi_projects"
RESUME = True
OVERWRITE = False
DEBUG_PRINT = False

# Variables
resolution_relative = 0.001
epsg_code = 3031

# Steps
STEPS = {
    "set_focal_length": False,
    "set_camera_location": False,
    "detect_fiducials": False,
    "union_masks": False,
    "match_photos": False,
    "align_cameras": False,
    "build_depth_maps": False,
    "build_dense_cloud": False,
    "build_mesh": False,
    "create_bounding_box": False,
    "build_dem": False,
    "build_orthomosaic": False,
    "create_gcps": False,
    "load_gcps": False,
    "export_alignment": True,
    "build_pointcloud": False
}

# Debug settings
DEBUG = {
    "save_thumbnails": False,
    "save_masks": False,
    "save_adapted_masks": False,
    "save_key_points": False,
    "save_tie_points": False,
}


def run_agi_relative(project_name, images,
                     camera_positions=None, camera_accuracies=None,
                     camera_rotations=None, camera_footprints=None,
                     focal_lengths=None):
    if RESUME and OVERWRITE:
        raise ValueError("Both RESUME and OVERWRITE cannot be set to True.")

    # define path to the project folder and the project files
    project_fld = os.path.join(PATH_PROJECT_FOLDERS, project_name)
    project_file_path_relative = project_fld + "/" + project_name + "_relative.psx"

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

    # set default values for mutable arguments
    if camera_positions is None:
        camera_positions = {}
    if camera_accuracies is None:
        camera_accuracies = {}
    if camera_rotations is None:
        camera_rotations = {}
    if camera_footprints is None:
        camera_footprints = {}
    if focal_lengths is None:
        focal_lengths = {}

    # init some path variables
    output_dem_path = None
    output_ortho_path = None

    # remove the complete project folder if OVERWRITE is set to True
    if OVERWRITE and os.path.exists(project_fld):
        print(f"Remove '{project_fld}'")
        shutil.rmtree(project_fld)

    # create a metashape project object
    doc = Metashape.Document(read_only=False)

    # create project folder if not existing
    if os.path.isdir(project_fld) is False:
        os.makedirs(project_fld)

    # check if the project already exists
    if os.path.exists(project_file_path_relative):

        if RESUME is False:
            raise FileExistsError("The project already exists. Set RESUME to True to resume the project.")

        # load the project
        doc.open(project_file_path_relative, ignore_lock=True)

    else:
        # save the project with file path so that later steps can be resumed
        doc.save(project_file_path_relative)

    # init output folder
    output_fld = os.path.join(project_fld, "output_relative")
    if os.path.isdir(output_fld) is False:
        os.makedirs(output_fld)

    # init data folder
    data_fld = os.path.join(project_fld, "data")
    if os.path.isdir(data_fld) is False:
        os.makedirs(data_fld)

    # init display folder
    display_fld = os.path.join(project_fld, "display")
    if os.path.isdir(display_fld) is False:
        os.makedirs(display_fld)

    # add a chunk
    if len(doc.chunks) == 0:
        chunk = doc.addChunk()

        group = chunk.addCameraGroup()
        group.type = Metashape.CameraGroup.Type.Folder

        # add the images to the chunk
        chunk.addPhotos(images)

        for camera in chunk.cameras:
            camera.group = group
            camera.sensor.film_camera = True
            camera.sensor.fixed_calibration = True

    else:
        chunk = doc.chunks[0]

    if DEBUG["save_thumbnails"]:
        thumb_folder = os.path.join(display_fld, "thumbnails")
        if not os.path.exists(thumb_folder):
            os.makedirs(thumb_folder)

        for camera in chunk.cameras:
            thumb_path = os.path.join(thumb_folder, f"{camera.label}_thumb.jpg")

            image = li.load_image(camera.label)
            et.export_thumbnail(image, thumb_path)

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

    if STEPS["detect_fiducials"]:

        print("Detect fiducials")

        # detect fiducials
        chunk.detectFiducials(generate_masks=True, generic_detector=False, frame_detector=True,
                              fiducials_position_corners=False)

        for camera in chunk.cameras:
            camera.sensor.calibrateFiducials(0.025)

        doc.save()

        print("Detect fiducials - finished")

    # save masks
    if DEBUG["save_masks"]:
        mask_folder = os.path.join(data_fld, "masks_original")
        if not os.path.exists(mask_folder):
            os.makedirs(mask_folder)

        for camera in chunk.cameras:
            if camera.mask is not None:
                mask_path = os.path.join(mask_folder, f"{camera.label}_mask.tif")
                camera.mask.image().save(mask_path)

    if STEPS["union_masks"]:

        print("Union masks")

        for camera in chunk.cameras:
            if camera.enabled and camera.mask:
                mask = camera.mask.image()

                m_width = mask.width
                m_height = mask.height

                # convert to np array
                mask_bytes = mask.tostring()
                existing_mask = np.frombuffer(mask_bytes, dtype=np.uint8).reshape((m_height, m_width))

                # get the image id
                image_id = camera.label.split("_")[0]

                # create an adapted mask
                adapted_mask = cam.create_adapted_mask(existing_mask, image_id)
                adapted_mask_m = Metashape.Image.fromstring(adapted_mask,
                                                            adapted_mask.shape[1],
                                                            adapted_mask.shape[0],
                                                            channels=' ',
                                                            datatype='U8')

                mask_obj = Metashape.Mask()
                mask_obj.setImage(adapted_mask_m)

                camera.mask = mask_obj

        doc.save()

        print("Union masks - finished")

    # save masks
    if DEBUG["save_adapted_masks"]:
        mask_folder = os.path.join(data_fld, "masks_adapted")
        if not os.path.exists(mask_folder):
            os.makedirs(mask_folder)

        for camera in chunk.cameras:
            if camera.mask is not None:
                mask_path = os.path.join(mask_folder, f"{camera.label}_mask_adapted.tif")
                camera.mask.image().save(mask_path)

    # match photos
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

    # save key points
    if DEBUG["save_key_points"]:

        # define save path
        kp_fld = os.path.join(display_fld, "key_points")
        if not os.path.exists(kp_fld):
            os.makedirs(kp_fld)

        # get image ids
        image_ids = [camera.label for camera in chunk.cameras]

        # call snippet to save key points
        skp.save_key_points(image_ids, project_fld, kp_fld)

    # align cameras
    if STEPS["align_cameras"]:
        print("Align cameras")

        # align cameras
        chunk.alignCameras(reset_alignment=True, adaptive_fitting=False)
        doc.save()

        print("Align cameras - finished")

    # save tie points
    if DEBUG["save_tie_points"]:
        tp_fld = os.path.join(display_fld, "tie_points")
        if not os.path.exists(tp_fld):
            os.makedirs(tp_fld)
        stp.save_tie_points(chunk, tp_fld)

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
        output_model_path = os.path.join(output_fld, project_name + "_model_relative.obj")

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
        min_corner = Metashape.Vector([center.x - size.x / 2, center.y - size.y / 2])
        max_corner = Metashape.Vector([center.x + size.x / 2, center.y + size.y / 2])

        # Create the bounding box
        bounding_box = Metashape.BBox(min_corner, max_corner)
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
            'resolution': resolution_relative
        }

        # add region to build parameters
        if bounding_box is not None:
            build_params['region'] = bounding_box

        # build the DEM
        chunk.buildDem(**build_params)
        doc.save()

        # define output path for the DEM
        output_dem_path = os.path.join(output_fld, project_name + "_dem_relative.tif")

        # check if the chunk is in relative mode
        print("TODO ADD RELATIVE CHECK FOR DEM")

        # set export parameters for the DEM
        export_params = {
            'path': output_dem_path,
            'source_data': Metashape.ElevationData,
            'image_format': Metashape.ImageFormatTIFF,
            'raster_transform': Metashape.RasterTransformNone,
            'nodata_value': -9999,
            'resolution': resolution_relative
        }
        if bounding_box is not None:
            export_params['region'] = bounding_box

        chunk.exportRaster(**export_params)

        print("Build DEM - finished")

    """
    # Accessing geolocation information for each camera (image)
    for camera in chunk.cameras:
        if camera.reference.location:  # Check if location data is available
            print("Camera:", camera.label)
            print("Location:", camera.reference.location)  # Prints Vector(x, y, z)
            print("Accuracy:", camera.reference.accuracy)  # Prints accuracy if available
        else:
            print("Camera:", camera.label, "has no geolocation data.")
    """

    # build ortho-mosaic
    if STEPS["build_orthomosaic"]:

        print("Build orthomosaic")

        projection = Metashape.OrthoProjection()
        projection.crs = chunk.crs

        build_params = {
            'surface_data': Metashape.ModelData,
            'blending_mode': Metashape.MosaicBlending,
            'projection': projection,
            'resolution': resolution_relative
        }
        if bounding_box is not None:
            build_params['region'] = bounding_box

        chunk.buildOrthomosaic(**build_params)
        doc.save()

        output_ortho_path = os.path.join(output_fld, project_name + "_ortho_relative.tif")

        # check if the chunk is in relative mode
        print("TODO ADD RELATIVE CHECK FOR ORTHO")

        export_params = {
            'path': output_ortho_path,
            'source_data': Metashape.OrthomosaicData,
            'image_format': Metashape.ImageFormatTIFF,
            'raster_transform': Metashape.RasterTransformNone,
            'nodata_value': -9999,
            'resolution': resolution_relative
        }
        if bounding_box is not None:
            export_params['region'] = bounding_box

        chunk.exportRaster(**export_params)

        print("Build orthomosaic - finished")

    if STEPS["create_gcps"]:

        # use default values for DEM and ortho if not given
        if output_dem_path is None:
            output_dem_path = os.path.join(output_fld, project_name + "_dem_relative.tif")
        if output_ortho_path is None:
            output_ortho_path = os.path.join(output_fld, project_name + "_ortho_relative.tif")

        # load the required data
        dem = li.load_image(output_dem_path)
        ortho = li.load_image(output_ortho_path)
        footprints = camera_footprints

        # define path to save gcp files
        gcp_path = os.path.join(data_fld, "gcps.csv")

        # call snippet to export gcps
        eg.export_gcps(dem, ortho, bounding_box, resolution_relative,
                       footprints, gcp_path)

    if STEPS["load_gcps"]:

        # load gcps from file
        gcp_path = os.path.join(data_fld, "gcps.csv")
        gcps = pd.read_csv(gcp_path, sep=';')

        # set crs of markers
        chunk.marker_crs = Metashape.CoordinateSystem(f"EPSG::{epsg_code}")  # noqa
        chunk.marker_location_accuracy = [25, 25, 100]

        # multiply the number of gcps by the number of cameras to get the total number of possible markers
        ttl = gcps.shape[0] * len(chunk.cameras)
        pbar = tqdm(total=ttl)

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
                pbar.update(1)
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

    if STEPS["export_alignment"]:
        pc_path = os.path.join(data_fld, "point_cloud.ply")
        chunk.exportPointCloud(pc_path, source_data=Metashape.TiePointsData)

        # Export camera calibration and orientation
        camera_path = os.path.join(data_fld, "cameras.txt")
        with open(camera_path, 'w') as f:
            for camera in chunk.cameras:
                if camera.transform:
                    line = camera.label + ',' + ','.join(map(str, np.asarray(camera.transform).flatten())) + '\n'
                    f.write(line)

        #matches_path = os.path.join(data_fld, "matches.csv")
        #with open(matches_path, 'w') as f:
        #    for point in chunk.point_cloud.points:
        #        for track in point.tracks:
        #            camera = chunk.cameras[track.camera_id]
        #            projection = track.projection
        #            f.write(f"{point.coord[0]},{point.coord[1]},{point.coord[2]},
        #                      {camera.label},{projection[0]},{projection[1]}\n")

    if STEPS["build_pointcloud"]:
        chunk.buildPointCloud()
        doc.save()

        output_pc_path = os.path.join(output_fld, project_name + "_pointcloud_relative.laz")

        export_params = {
            'path': output_pc_path,
        }
        chunk.exportPointCloud(**export_params)

        doc.save()


if __name__ == "__main__":
    sys_project_name = sys.argv[1]
    sys_images = json.loads(sys.argv[2])
    sys_camera_positions = json.loads(sys.argv[3])
    sys_camera_accuracies = json.loads(sys.argv[4])
    sys_focal_lengths = json.loads(sys.argv[5])
    run_agi_relative(sys_project_name,
                     sys_images, sys_camera_positions,
                     sys_camera_accuracies, sys_focal_lengths)
