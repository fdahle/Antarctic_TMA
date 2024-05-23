import os

from Metashape import Metashape

PATH_PROJECT_FOLDER = "/data_1/ATM/data_1/sfm/agi_projects"
project_name = "test"

def init_agi(project_name, images):

    # create path to the project
    project_path = os.path.join(PATH_PROJECT_FOLDER, project_name + ".psx")

    # create a new metashape project
    doc = Metashape.Document()

    # save the project
    #doc.save(project_path)

    # add a chunk
    chunk = doc.addChunk()

    # add the images to the chunk
    chunk.addPhotos(images)

    for camera in chunk.cameras:
        print(camera.label)
        print(camera.photo.path)

    chunk.matchPhotos(generic_preselection=True, reference_preselection=False)
    chunk.alignCameras()

    for camera in chunk.cameras:
        print(camera.transform)

    export_path = os.path.join(PATH_PROJECT_FOLDER, project_name + ".las")
    chunk.exportPointCloud(path=export_path)

if __name__ == "__main__":

    images = ["CA196532V0010", "CA196532V0011", "CA196532V0012",
              "CA196532V0013", "CA196532V0014", "CA196532V0015",
              "CA196532V0016", "CA196532V0017", "CA196532V0018",
              "CA196532V0019", "CA196532V0020"]

    # get only the first 3 images
    images = images[:3]

    path_image_folder = "/data_1/ATM/data_1/aerial/TMA/downloaded"

    # create lst with absolute paths
    images = [os.path.join(path_image_folder, image + ".tif") for image in images]

    init_agi(project_name, images)