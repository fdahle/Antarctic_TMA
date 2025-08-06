import shutil
import time
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import warnings

# --- CONFIG ---
source_folder = Path("/mnt/Webdrive/staff-umbrella/vrlabarchive01/felix")
Image.MAX_IMAGE_PIXELS = None  # Disable decompression bomb limit
warnings.simplefilter('ignore', Image.DecompressionBombWarning)
# --------------

def safe_move(src, dst, retries=3, delay=2):
    for attempt in range(retries):
        try:
            shutil.move(str(src), str(dst))
            return True
        except Exception as e:
            print(f"Move failed for {src.name} (attempt {attempt+1}): {e}")
            time.sleep(delay)
    return False

# 1. Get all .tif files starting with 'CA'
tif_files = list(source_folder.glob("CA*.tif"))

for tif_path in tqdm(tif_files, desc="Processing TIFF files", unit="file"):
    try:
        with Image.open(tif_path) as img:
            img.verify()
    except Exception:
        print(f"\nDeleting corrupted file: {tif_path.name}")
        try:
            tif_path.unlink()
        except Exception as e:
            print(f"Failed to delete {tif_path.name}: {e}")
        continue

    # 2. Extract only the first 6 characters (e.g., "CA1234")
    folder_name = tif_path.name[:6]
    target_folder = source_folder / folder_name
    target_folder.mkdir(exist_ok=True)

    # 3. Move file into the new folder
    target_path = target_folder / tif_path.name
    if not safe_move(tif_path, target_path):
        print(f"Failed to move {tif_path.name} after retries.")
