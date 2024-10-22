import os
import Metashape

project_fld = "/data/ATM/data_1/sfm/agi_projects/"
project_name = "agi_agi"

# create the path to the project
project_path = os.path.join(project_fld, project_name, project_name + ".psx")

doc = Metashape.Document(read_only=False)  # noqa
doc.open(project_path, ignore_lock=True)
chunk = doc.chunk

for camera in chunk.cameras:
    print(camera.label, camera.transform)
    print(camera.reference.location)

path_file = ("/data/ATM/data_1/sfm/agi_projects/agi_agi/agi_agi.files/"
             "0/0/point_cloud/points0/c0.kpt")


import struct

with open(path_file, 'rb') as f:
    while True:
        # Each keypoint has an x, y coordinate, each a 4-byte float
        bytes = f.read(8)  # 4 bytes for x, 4 bytes for y
        if not bytes:
            break  # End of file
        x, y = struct.unpack('ff', bytes)  # 'ff' means two 4-byte floats
        print(f'Keypoint: x={x}, y={y}')
