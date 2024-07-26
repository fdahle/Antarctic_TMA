# Library imports
import os
os.environ['KMP_WARNINGS'] = '0'  # noqa: E402
import Metashape
import numpy as np
import shutil
import sys
import json

# Local imports
import src.base.load_credentials as lc
import src.load.load_image as li
import src.export.export_thumbnail as et
import src.sfm_agi.snippets.create_adapted_mask as cam
import src.sfm_agi.snippets.save_key_points as sk
import src.sfm_agi.snippets.save_tie_points as st

# Constants
PATH_PROJECT_FOLDERS = "/data_1/ATM/data_1/sfm/agi_projects"
RESUME = False
OVERWRITE = True

# Variables
film_cameras = True

# Steps
STEPS = {
    "set_focal_length": True,
    "set_camera_location": True,
    "detect_fiducials": True,
    "union_masks": True,
    "match_photos": True,
    "align_cameras": True,
    "build_depth_maps": True,
    "build_dense_cloud": True,
    "build_mesh": True,
    "build_dem": True,
    "build_orthomosaic": True,
    "build_pointcloud": True
}

# Debug settings
DEBUG = {
    "save_thumbnails": True,
    "save_masks": True,
    "save_key_points": True,
    "save_tie_points": True,
    "save_positions": True,
    "save_footprints": True
}


def init_agi(project_name, images,
             camera_positions=None, camera_accuracies=None,
             camera_rotations=None,
             focal_lengths=None, epsg_code=3031):

    print("STEPS:")
    print(STEPS)

    print("images:")
    print(images)

    if camera_positions is not None:
        print("camera_positions:")
        print(camera_positions)

    if camera_accuracies is not None:
        print("camera_accuracies:")
        print(camera_accuracies)

    if camera_rotations is not None:
        print("camera_rotations:")
        print(camera_rotations)

    if focal_lengths is not None:
        print("focal_lengths:")
        print(focal_lengths)

    # enable use of gpu
    Metashape.app.gpu_mask = 1
    Metashape.app.cpu_enable = False

    if RESUME and OVERWRITE:
        raise ValueError("Both RESUME and OVERWRITE cannot be set to True.")

    # get the license key
    licence_key = lc.load_credentials("agisoft")['licence']

    # Activate the license
    Metashape.License().activate(licence_key)

    # create a metashape project object
    doc = Metashape.Document(read_only=False)

    # set default values for mutable arguments
    if camera_positions is None:
        camera_positions = {}
    if camera_accuracies is None:
        camera_accuracies = {}
    if camera_rotations is None:
        camera_rotations = {}
    if focal_lengths is None:
        focal_lengths = {}

    # define path to the project folder and the project file
    project_fld = os.path.join(PATH_PROJECT_FOLDERS, project_name)
    project_file_path = project_fld + "/" + project_name + ".psx"

    # remove the project file if OVERWRITE is set to True
    if OVERWRITE and os.path.exists(project_fld):
        print(f"Remove '{project_fld}'")
        shutil.rmtree(project_fld)

    # check if the project already exists
    if os.path.exists(project_fld):

        if RESUME is False:
            raise FileExistsError("The project already exists. Set RESUME to True to resume the project.")

        # load the project
        doc.open(project_file_path, ignore_lock=True)
    else:
        print("Create folder at ", project_fld)
        os.makedirs(project_fld)

    # save the project with file path so that later steps can be resumed
    doc.save(project_file_path)

    # add a chunk
    if len(doc.chunks) == 0:
        chunk = doc.addChunk()

        if film_cameras:
            group = chunk.addCameraGroup()
            group.type = Metashape.CameraGroup.Type.Folder
        else:
            group = None

        # add the images to the chunk
        chunk.addPhotos(images)

        # Set every camera to film camera
        if film_cameras:

            for camera in chunk.cameras:
                camera.group = group
                camera.sensor.film_camera = True
                camera.sensor.fixed_calibration = True
                #camera.sensor.height = 0
                #camera.sensor.width = 0

    else:
        chunk = doc.chunks[0]

    if DEBUG["save_thumbnails"]:
        thumb_folder = os.path.join(project_fld, "thumbnails")
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

    if STEPS["detect_fiducials"] and film_cameras:

        print("Detect fiducials")

        # detect fiducials
        chunk.detectFiducials(generate_masks=True, generic_detector=False, frame_detector=True,
                              fiducials_position_corners=False)
        doc.save()

        for camera in chunk.cameras:
            camera.sensor.calibrateFiducials(0.025)

        doc.save()

        print("Detect fiducials - finished")

    if STEPS["set_camera_location"]:

        print("Set Camera location")

        # set the coordinate system of the chunk
        chunk.crs = Metashape.CoordinateSystem(f"EPSG::{epsg_code}")

        # set camera position if given
        for camera in chunk.cameras:
            if camera.label in camera_positions['image_id'].values:
                camera_row = camera_positions[camera_positions['image_id'] == camera.label].iloc[0]
                x, y, z = camera_row['x'], camera_row['y'], camera_row['z']
                print("Set camera location for", camera.label, "to", x, y, z)
                camera.reference.location = Metashape.Vector([x, y, z])

                # set the accuracy of the position if given
                if camera.label in camera_accuracies:
                    accuracy = camera_accuracies[camera.label]
                    camera.reference.accuracy = Metashape.Vector([accuracy[0], accuracy[1], accuracy[2]])

                # set the rotation of the camera if given
                if camera.label in camera_rotations.keys():
                    entry = camera_rotations[camera.label]
                    yaw, pitch, roll = entry[0], entry[1], entry[2]
                    camera.reference.rotation = Metashape.Vector([yaw, pitch, roll])
            else:
                print(f"WARNING: Camera position not given for {camera.label}")
        print("Set Camera location - finished")

    # save masks
    if DEBUG["save_masks"]:
        mask_folder = os.path.join(project_fld, "masks")
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
                                                            adapted_mask.shape[0], ' ',
                                                            datatype='U8')
                mask_obj = Metashape.Mask()
                mask_obj.setImage(adapted_mask_m)

                camera.mask = mask_obj

        doc.save()

        print("Union masks - finished")

    # save masks
    if DEBUG["save_masks"]:
        mask_folder = os.path.join(project_fld, "masks")
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
        save_path = os.path.join(project_fld, "key_points")

        # create folder if not exists
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # get image ids
        image_ids = [camera.label for camera in chunk.cameras]

        # call snippet to save key points
        sk.save_key_points(image_ids, project_fld)

    # align cameras
    if STEPS["align_cameras"]:

        print("Align cameras")

        # align cameras
        chunk.alignCameras(reset_alignment=True, adaptive_fitting=False)
        doc.save()

        print("Align cameras - finished")

    # save tie points
    if DEBUG["save_tie_points"]:
        save_fld = os.path.join(project_fld, "tie_points")
        if not os.path.exists(save_fld):
            os.makedirs(save_fld)
        st.save_tie_points(chunk, save_fld)

    import geopandas as gpd
    from shapely.geometry import Point, Polygon

    if DEBUG["save_positions"]:
        save_fld = os.path.join(project_fld, "shapes")
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

        # define output folder
        output_fld = os.path.join(project_fld, "output")
        if os.path.isdir(output_fld) is False:
            os.makedirs(output_fld)

        # define output path for the model
        output_model_path = os.path.join(output_fld, project_name + "_model.obj")

        # define export parameters
        export_params = {
            'path': output_model_path,
        }

        # export the model
        chunk.exportModel(**export_params)

        print("Build mesh - finished")

    # build DEM
    if STEPS["build_dem"]:

        print("Build DEM")

        projection = Metashape.OrthoProjection()
        projection.crs = chunk.crs

        chunk.buildDem(source_data=Metashape.DataSource.PointCloudData,
                       interpolation=Metashape.Interpolation.EnabledInterpolation,
                       projection=projection)
        doc.save()

        output_fld = os.path.join(project_fld, "output")
        if os.path.isdir(output_fld) is False:
            os.makedirs(output_fld)

        output_dem_path = os.path.join(output_fld, project_name + "_dem.tif")

        export_params = {
            'path': output_dem_path,
            'source_data': Metashape.ElevationData,
            'image_format': Metashape.ImageFormatTIFF,
            'raster_transform': Metashape.RasterTransformNone,
            'nodata_value': -9999,
            'resolution': 10
        }

        chunk.exportRaster(**export_params)

        print("Build DEM - finished")

    # Accessing geolocation information for each camera (image)
    for camera in chunk.cameras:
        if camera.reference.location:  # Check if location data is available
            print("Camera:", camera.label)
            print("Location:", camera.reference.location)  # Prints Vector(x, y, z)
            print("Accuracy:", camera.reference.accuracy)  # Prints accuracy if available
        else:
            print("Camera:", camera.label, "has no geolocation data.")

    # build ortho-mosaic
    if STEPS["build_orthomosaic"]:

        print("Build orthomosaic")

        projection = Metashape.OrthoProjection()
        projection.crs = chunk.crs

        chunk.buildOrthomosaic(surface_data=Metashape.ModelData,
                               blending_mode=Metashape.MosaicBlending)
                               #projection=projection)
        doc.save()

        output_fld = os.path.join(project_fld, "output")
        if os.path.isdir(output_fld) is False:
            os.makedirs(output_fld)

        output_ortho_path = os.path.join(output_fld, project_name + "_ortho.tif")

        export_params = {
            'path': output_ortho_path,
            'source_data': Metashape.OrthomosaicData,
            'image_format': Metashape.ImageFormatTIFF,
            'raster_transform': Metashape.RasterTransformNone,
            'nodata_value': -9999,
            #'resolution': 10
        }
        chunk.exportRaster(**export_params)

        print("Build orthomosaic - finished")

    if STEPS["build_pointcloud"]:

        chunk.buildPointCloud()
        doc.save()

        output_fld = os.path.join(project_fld, "output")
        if os.path.isdir(output_fld) is False:
            os.makedirs(output_fld)

        output_pc_path = os.path.join(output_fld, project_name + "_pointcloud.laz")

        export_params = {
            'path': output_pc_path,
        }
        chunk.exportPointCloud(**export_params)

    if DEBUG["save_footprints"]:

        save_fld = os.path.join(project_fld, "shapes")
        if not os.path.exists(save_fld):
            os.makedirs(save_fld)

        #save_path = os.path.join(save_fld, "footprints.shp")

        import src.sfm_agi.snippets.create_footprints as cf
        cf.create_footprints(chunk, save_fld)


if __name__ == "__main__":

    sys_project_name = sys.argv[1]
    sys_images = json.loads(sys.argv[2])
    sys_camera_positions = json.loads(sys.argv[3])
    sys_camera_accuracies = json.loads(sys.argv[4])
    sys_focal_lengths = json.loads(sys.argv[5])
    init_agi(sys_project_name,
             sys_images, sys_camera_positions,
             sys_camera_accuracies, sys_focal_lengths)
