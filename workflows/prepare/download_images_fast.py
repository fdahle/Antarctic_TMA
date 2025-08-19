import gzip
import os
import shutil
import ssl
import tempfile
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

import src.other.misc.compress_tif_file as ctf

# Constants
PATH_DOWNLOAD_FLD = "/media/fdahle/d3f2d1f5-52c3-4464-9142-3ad7ab1ec06d/data_1/aerial/TMA/downloaded"
PATH_FINAL_FLD = "/mnt/Webdrive/staff-umbrella/vrlabarchive01/felix"
PATH_TEMP_FLD = "/data/ATM/temp"
USGS_URL = "http://data.pgc.umn.edu/aerial/usgs/tma/photos/high"
MAX_WORKERS = 8  # Adjust for safety against server bans

# Allow insecure HTTPS (needed for some university networks or outdated SSL configs)
ssl._create_default_https_context = ssl._create_unverified_context

def download_image_from_usgs(image_id: str,
                             session: requests.Session,
                             final_folder: str = PATH_FINAL_FLD,
                             url_image: str = USGS_URL,
                             overwrite: bool = False,
                             verbose: bool = False) -> bool:
    """
    Download and decompress a TMA image from the USGS archive using requests.
    """

    # Parse image components
    tma = image_id[0:6]
    direction = image_id[6:9]
    roll = image_id[9:13]

    url = f"{url_image}/{tma}/{tma}{direction}/{tma}{direction}{roll}.tif.gz"

    final_path = os.path.join(final_folder, f"{image_id}.tif")
    if not overwrite and os.path.isfile(final_path):
        if verbose:
            print(f"{image_id} already exists at final location.")
        return True

    try:
        # Ensure output folder exists
        os.makedirs(final_folder, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            gz_path = os.path.join(tmpdir, f"{image_id}.tif.gz")
            img_path = os.path.join(tmpdir, f"{image_id}.tif")

            with session.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(gz_path, 'wb') as f:
                    shutil.copyfileobj(r.raw, f)

            # Decompress .gz
            with gzip.open(gz_path, 'rb') as f_in, open(img_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

            # Compress .tif
            status = ctf.compress_tif_file(img_path, compression_method='lzw', quality=100)
            if not status:
                print(f"Compression failed for {image_id}.")
                return False

            # Move to final destination
            shutil.move(img_path, final_path)

        if verbose:
            print(f"{image_id} downloaded and moved successfully.")
        return True

    except Exception as e:
        print(f"Download failed for {image_id}: {e}")
        return False

import glob
def list_existing_tifs_recursive(*roots) -> set[str]:
    """Return basenames (no .tif) of all .tif files under given roots, recursively."""
    names = set()
    for root in roots:
        if not os.path.isdir(root):
            continue
        for p in glob.iglob(os.path.join(root, "**", "*.tif"), recursive=True):
            names.add(os.path.splitext(os.path.basename(p))[0])
    return names


if __name__ == "__main__":
    # Load shapefile and construct image IDs
    import src.load.load_shape_data as lsd
    pth_photos = "/data/ATM/data_1/shapefiles/TMA_Photocenters/TMA_pts_20100927.shp"
    df = lsd.load_shape_data(pth_photos)
    print("Shapefile loaded with", len(df), "entries.")

    existing = list_existing_tifs_recursive(PATH_DOWNLOAD_FLD,
                                            PATH_TEMP_FLD, PATH_FINAL_FLD)

    print(len(existing), "existing images found.")

    mapping = {'L': '31L', 'V': '32V', 'R': '33R'}
    image_ids = [
        f"{eid[:6]}{mapping[letter]}{eid[6:]}"
        for eid, views in zip(df['ENTITY_ID'], df['VIEWS'])
        for letter in views
    ]

    # Filter out already downloaded files
    image_ids = [img_id for img_id in image_ids if img_id not in existing]

    import random
    random.shuffle(image_ids)

    session = requests.Session()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [
            executor.submit(download_image_from_usgs, image_id, session)
            for image_id in image_ids
        ]
        for _ in tqdm(as_completed(futures), total=len(futures)):
            pass
