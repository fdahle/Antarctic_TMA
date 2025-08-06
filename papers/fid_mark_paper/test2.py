import glob
import os
import csv
os.environ['KMP_WARNINGS'] = '0'

target_folder = "/home/fdahle/Desktop/fid_marks"
image_paths = glob.glob(os.path.join(target_folder, "*.tif"))

# Load existing entries if CSV exists
csv_path = "/home/fdahle/Desktop/fiducial_coords.csv"
existing_ids = set()

if os.path.exists(csv_path):
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            existing_ids.add(row["image_id"] + ".tif")

print(len(image_paths))

image_paths = [target_folder + "/" + "CA214033R0463" + ".tif"]

# Filter out already processed images
image_paths = [p for p in image_paths if os.path.basename(p) not in existing_ids]


print(len(image_paths))

# limit list to 10 images for testing
image_paths = image_paths[:250]  # Adjust the number as needed

if not image_paths:
    print("No new images to process.")
    exit()

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
chunk.addPhotos(image_paths)

for camera in chunk.cameras:
    camera.group = group
    camera.sensor.film_camera = True
    camera.sensor.fixed_calibration = True
    camera.sensor.focal_length = 151.982

from tqdm import tqdm
def create_progress_bar(desc: str = "Processing", total: int = 100):
    """
    Initializes a tqdm progress bar and returns a callback for Metashape progress tracking.
    """
    pbar = tqdm(total=total, desc=desc, unit="%",
                bar_format="{desc}: {percentage:3.0f}% |{bar}|"
                           " [{elapsed}<{remaining}, {rate_fmt}]",
                position=0, leave=True)

    def progress_callback(progress: float):
        pbar.n = int(progress)
        pbar.refresh()
        if pbar.n >= total:
            pbar.close()

    return progress_callback


chunk.detectFiducials(generate_masks=False, generic_detector=False, frame_detector=True, fiducials_position_corners=False,
                      progress=create_progress_bar("Detect Fiducials"))

label_map = {"__auto_4": "N", "__auto_5": "E", "__auto_6": "S", "__auto_7": "W"}

import csv
csv_path = "/home/fdahle/Desktop/fiducial_coords.csv"
# build a dict of just the fiducial markers we care about

from tqdm import tqdm
file_exists = os.path.exists(csv_path)
with open(csv_path, "a", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=["image_id", "N", "E", "S", "W"])
    if not file_exists:
        writer.writeheader()

    for camera in tqdm(chunk.cameras):
        cam_label = camera.label
        fid_present = {key: False for key in ["N", "E", "S", "W"]}

        for marker in chunk.markers:
            if marker.type != Metashape.Marker.Type.Fiducial:
                continue

            for cam, proj in marker.projections.items():

                if cam.label == cam_label:
                    mapped_label = label_map.get(marker.label)
                    if mapped_label:
                        fid_present[mapped_label] = True


        # Write to CSV
        writer.writerow({
            "image_id": cam_label,
            "N": fid_present["N"],
            "E": fid_present["E"],
            "S": fid_present["S"],
            "W": fid_present["W"]
        })
