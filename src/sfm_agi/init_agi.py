import os

import Metashape

PATH_PROJECT_FOLDERS = "/data_1/ATM/data_1/sfm/agi_projects"


def init_agi(project_name, images,
             camera_positions=None, camera_accuracies=None,
             focal_lengths=None):
    # set default values for mutable arguments
    if camera_positions is None:
        camera_positions = {}
    if camera_accuracies is None:
        camera_accuracies = {}
    if focal_lengths is None:
        focal_lengths = {}

    # create path to the project
    project_fld = os.path.join(PATH_PROJECT_FOLDERS, project_name)

    # create the project folder if it does not exist
    if not os.path.exists(project_fld):
        os.makedirs(project_fld)

    project_file_path = project_fld + "/" + project_name + ".psx"

    # create a new metashape project
    doc = Metashape.Document()

    # save the project
    # doc.save(project_file_path)

    # add a chunk
    chunk = doc.addChunk()

    # set the coordinate system of the chunk
    chunk.crs = Metashape.CoordinateSystem("EPSG::3031")

    # add the images to the chunk
    chunk.addPhotos(images)

    # Set some settings for each camera
    for camera in chunk.cameras:

        # set to film camera
        camera.sensor.film_camera = True

        # set focal length if given
        if camera.label in focal_lengths:
            focal_length = focal_lengths[camera.label]
            camera.sensor.focal_length = focal_length

        # set camera position if given
        if camera.label in camera_positions:
            x, y, z = camera_positions[camera.label]
            camera.reference.location = Metashape.Vector([x, y, z])

            # set the accuracy of the position if given
            if camera.label in camera_accuracies:
                accuracy = camera_accuracies[camera.label]
                camera.reference.accuracy = Metashape.Vector([accuracy[0], accuracy[1], accuracy[2]])

    chunk.detectFiducials(generate_masks=True, cameras=chunk.cameras)
    print("Found the following fiducials in the images: ")
    for marker in chunk.markers:
        print(marker.label)

    # save masks
    mask_folder = os.path.join(project_fld, "masks")
    for camera in chunk.cameras:
        mask_path = os.path.join(mask_folder, f"{camera.label}_mask.tif")
        camera.mask.image().save(mask_path)

    # match photos
    chunk.matchPhotos(generic_preselection=True, reference_preselection=False)

    _save_tps(chunk)

    # align cameras
    chunk.alignCameras()

    # build depth maps
    chunk.buildDepthMaps(quality=Metashape.HighQuality, filter_mode=Metashape.MildFiltering)

    # build dense cloud
    chunk.buildDenseCloud()

    # build mesh
    chunk.buildModel(surface_type=Metashape.Arbitrary)

    # build DEM
    chunk.buildDem(source=Metashape.DataSource.DenseCloudData)

    # build ortho-mosaic
    chunk.buildOrthomosaic(surface_data=Metashape.ModelData, blending_mode=Metashape.MosaicBlending)

    # export_path = os.path.join(PATH_PROJECT_FOLDER, project_name + ".las")
    # chunk.exportPointCloud(path=export_path)


def _save_tps(chunk):
    import numpy as np
    import src.display.display_images as di
    import src.load.load_image as li

    # Create a dictionary to map camera labels to file paths
    camera_file_paths = {camera.label: camera.photo.path for camera in chunk.cameras}

    tie_points_dict = {}

    # Access tie points directly
    tie_points = chunk.tie_points

    for tie_point in tie_points.points:
        if not tie_point.valid:
            continue

        projections = tie_point.projections
        for i, proj1 in enumerate(projections):
            for j, proj2 in enumerate(projections):
                if i >= j:
                    continue

                camera1 = proj1.camera
                camera2 = proj2.camera

                coord1 = proj1.coord
                coord2 = proj2.coord

                if (camera1.label, camera2.label) not in tie_points_dict:
                    tie_points_dict[(camera1.label, camera2.label)] = []

                tie_points_dict[(camera1.label, camera2.label)].append([coord1.x, coord1.y, coord2.x, coord2.y])

    # Convert lists to numpy arrays
    for key in tie_points_dict:
        tie_points_dict[key] = np.array(tie_points_dict[key])

    for (camera1_label, camera2_label), points in tie_points_dict.items():
        image1_path = camera_file_paths[camera1_label]
        image2_path = camera_file_paths[camera2_label]
        image1 = li.load_image(image1_path)
        image2 = li.load_image(image2_path)
        print(f"Tie points between {camera1_label} and {camera2_label}:")
        di.display_images([image1, image2], tie_points=points)
