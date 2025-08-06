import glob
import os
from collections import defaultdict
import shutil
from tqdm import tqdm

"""
flight_paths = ["1684", "1813", "1816", "1821", "1822", "1824", "1825", "1826", "1827",
                "1833", "1846", "2073", "2075", "2136", "2137", "2139", "2140", "2141", "2142", "2143"]
fld_images = "/data/ATM/data_1/aerial/TMA/downloaded"
fld_images2 = "/media/fdahle/d3f2d1f5-52c3-4464-9142-3ad7ab1ec06d/data_1/aerial/TMA/downloaded"

# Store counts in a nested dictionary: {flight_path: {'L': x, 'V': y, 'R': z}}
image_counts = defaultdict(lambda: {'L': 0, 'V': 0, 'R': 0})
unique_image_paths = {}

images = []
seen = set()

for fp in flight_paths:
    # Get all matching files from both folders
    files1 = glob.glob(os.path.join(fld_images, f"CA{fp}*.tif"))
    files2 = glob.glob(os.path.join(fld_images2, f"CA{fp}*.tif"))

    # Prioritize files1 (primary), fall back to files2 if not already seen
    for f in files1 + files2:
        fname = os.path.basename(f)
        if fname not in seen:
            seen.add(fname)
            images.append(f)

            # Count by camera type
            if 'L' in fname:
                image_counts[fp]['L'] += 1
            elif 'V' in fname:
                image_counts[fp]['V'] += 1
            elif 'R' in fname:
                image_counts[fp]['R'] += 1

# Print results
for fp in flight_paths:
    counts = image_counts[fp]
    total = counts['L'] + counts['V'] + counts['R']
    print(f"Flight path {fp}: Total={total}, L={counts['L']}, V={counts['V']}, R={counts['R']}")

target_folder = "/home/fdahle/Desktop/fid_marks"
os.makedirs(target_folder, exist_ok=True)  # ensure it exists

for src_path in tqdm(images, desc="Copying images"):
    filename = os.path.basename(src_path)
    dst_path = os.path.join(target_folder, filename)

    if not os.path.exists(dst_path):
        shutil.copy2(src_path, dst_path)
"""
exit()

target_folder = "/home/fdahle/Desktop/fid_marks"
image_paths = glob.glob(os.path.join(target_folder, "*.tif"))


import Metashape
import src.base.load_credentials as lc

# get the license key
licence_key = lc.load_credentials("agisoft")['licence']

# Activate the license
Metashape.License().activate(licence_key)

# create a metashape project object
doc = Metashape.Document(read_only=False)

# add chunk
chunk = doc.addChunk()

# add camera group
group = chunk.addCameraGroup()
group.type = Metashape.CameraGroup.Type.Folder
# add cameras
chunk.addPhotos(images)

for camera in chunk.cameras:
    camera.group = group
    camera.sensor.film_camera = True
    camera.sensor.fixed_calibration = True
    camera.sensor.focal_length = 151.982

chunk.detectFiducials(generate_masks=False, generic_detector=False, frame_detector=True, fiducials_position_corners=False)
