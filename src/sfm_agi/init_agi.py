import os
os.environ['KMP_WARNINGS'] = '0'

# Library imports
import Metashape
import numpy as np

# Local imports
import src.base.load_credentials as lc
import src.sfm_agi.snippets.create_text_mask as ctm
import src.sfm_agi.snippets.save_key_points as sk
import src.sfm_agi.snippets.save_tie_points as st

PATH_PROJECT_FOLDERS = "/data_1/ATM/data_1/sfm/agi_projects"
RESUME = True
OVERWRITE = False

STEPS = {
    "set_camera_attributes": False,
    "detect_fiducials": False,
    "mask_text": False,
    "match_photos": False,
    "align_cameras": True,
    "build_depth_maps": True,
    "build_dense_cloud": True,
    "build_mesh": True,
    "build_dem": True,
    "build_orthomosaic": True,
    "export_point_cloud": True
}

DEBUG = {
    "save thumbnails": False,
    "save_masks": False,
    "save_key_points": False,
    "save_tie_points": True
}

def init_agi(project_name, images,
             camera_positions=None, camera_accuracies=None,
             focal_lengths=None):

    print(Metashape.app.settings.value("teak_name"))
    exit()

    # Metashape.app.cpu_enable = True

    if RESUME and OVERWRITE:
        raise ValueError("Both RESUME and OVERWRITE cannot be set to True.")

    if OVERWRITE:
        # set all values in STEPS to True
        for key in STEPS:
            STEPS[key] = True

    # get the license key
    licence_key = lc.load_credentials("agisoft")['licence']

    # Activate the license
    Metashape.License().activate(licence_key)

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

    # define the path to the project file
    project_file_path = project_fld + "/" + project_name + ".psx"

    # create a metashape project object
    doc = Metashape.Document(read_only=False)

    # remove the project file if OVERWRITE is set to True
    if OVERWRITE:
        print(f"Remove '{project_file_path}'")
        if os.path.exists(project_file_path):
            os.remove(project_file_path)

    # check if the project already exists
    if os.path.exists(project_file_path):

        if RESUME is False:
            raise FileExistsError("The project already exists. Set RESUME to True to resume the project.")

        # load the project
        doc.open(project_file_path, ignore_lock=True)

    print(f"{len(doc.chunks)} chunks found in the project.")

    # add a chunk
    if len(doc.chunks) == 0:
        chunk = doc.addChunk()

        # set the coordinate system of the chunk
        chunk.crs = Metashape.CoordinateSystem("EPSG::3031")

        # add the images to the chunk
        chunk.addPhotos(images)

    else:
        chunk = doc.chunks[0]

    if STEPS["set_camera_attributes"]:

        print("Set Camera attributes")

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

    if STEPS["detect_fiducials"]:

        print("Detect fiducials")

        chunk.detectFiducials(generate_masks=True, cameras=chunk.cameras)

    # save masks
    if DEBUG["save_masks"]:
        mask_folder = os.path.join(project_fld, "masks")
        if not os.path.exists(mask_folder):
            os.makedirs(mask_folder)

        for camera in chunk.cameras:
            mask_path = os.path.join(mask_folder, f"{camera.label}_mask.tif")
            camera.mask.image().save(mask_path)

    if STEPS["mask_text"]:

        print("Mask text")

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

                # create a text mask
                text_mask = ctm.create_text_mask(mask.width, mask.height, image_id)
                text_mask = text_mask.astype(np.uint8)

                new_mask = np.minimum(existing_mask, text_mask)
                new_mask = Metashape.Image.fromstring(new_mask, new_mask.shape[1], new_mask.shape[0], ' ',
                                                      datatype='U8')
                mask_obj = Metashape.Mask()
                mask_obj.setImage(new_mask)
                # mask2.setImage(metashape_image)

                camera.mask = mask_obj

    # save masks
    if DEBUG["save_masks"]:
        mask_folder = os.path.join(project_fld, "masks")
        if not os.path.exists(mask_folder):
            os.makedirs(mask_folder)

        for camera in chunk.cameras:
            mask_path = os.path.join(mask_folder, f"{camera.label}_mask_adapted.tif")
            camera.mask.image().save(mask_path)

    if STEPS["match_photos"]:

        print("Match photos")

        # create pairs from the cameras
        pairs = []
        num_cameras = len(chunk.cameras)
        for i in range(num_cameras - 1):
            pairs.append((i, i + 1))

        # match photos
        chunk.matchPhotos(generic_preselection=True, reference_preselection=False,
                          reference_preselection_mode=Metashape.ReferencePreselectionMode.ReferencePreselectionEstimated,
                          pairs=pairs,
                          filter_mask=True, mask_tiepoints=True,
                          filter_stationary_points=True, reset_matches=True)

    if DEBUG["save_key_points"]:

        # project must be saved to extract the key points
        doc.save(project_file_path)

        # define save path
        save_path = os.path.join(project_fld, "key_points")

        # create folder if not exists
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        image_ids = [camera.label for camera in chunk.cameras]
        sk.save_key_points(image_ids, project_fld)

    # align cameras
    if STEPS["align_cameras"]:

        print("Align cameras")

        chunk.alignCameras(reset_alignment=True, adaptive_fitting=True)

    if DEBUG["save_tie_points"]:
        save_path = os.path.join(project_fld, "tie_points")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        st.save_tie_points(chunk, save_path)

    doc.save(project_file_path)

    # build depth maps
    if STEPS["build_depth_maps"]:

        print("Build depth maps")
        chunk.buildDepthMaps()

    # build dense cloud
    if STEPS["build_dense_cloud"]:

        print("Build dense cloud")

        chunk.buildPointCloud()

    # build mesh
    if STEPS["build_mesh"]:

        print("Build mesh")

        chunk.buildModel(surface_type=Metashape.Arbitrary)

    # build DEM
    if STEPS["build_dem"]:

        print("Build DEM")
        chunk.buildDem(source=Metashape.DataSource.DenseCloudData)

    # build ortho-mosaic
    if STEPS["build_orthomosaic"]:

        print("Build orthomosaic")
        chunk.buildOrthomosaic(surface_data=Metashape.ModelData, blending_mode=Metashape.MosaicBlending)

    # export_path = os.path.join(PATH_PROJECT_FOLDER, project_name + ".las")
    # chunk.exportPointCloud(path=export_path)


def _save_thumbnails(images):

    for image in images:
        print(image)
