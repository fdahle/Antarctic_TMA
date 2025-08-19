import os
import shutil
from tqdm import tqdm


def move_files(src_folder, dest_folder):
    # Check if source and destination folders exist
    if not os.path.exists(src_folder):
        print(f"Source folder '{src_folder}' does not exist.")
        return

    if not os.path.exists(dest_folder):
        print(f"Destination folder '{dest_folder}' does not exist.")
        return

    # List all TIFF files in the source folder
    tiff_files = [f for f in os.listdir(src_folder) if f.lower().endswith('.tif')]

    # Total files to move
    total_files = len(tiff_files)

    # Initialize tqdm progress bar
    with tqdm(total=total_files, desc="Moving TIFF files", unit="file") as pbar:
        # Loop through the TIFF files and move them
        for filename in tiff_files:
            src_file = os.path.join(src_folder, filename)
            dest_file = os.path.join(dest_folder, filename)

            # Ensure it's a file (not a directory)
            if os.path.isfile(src_file):
                try:
                    shutil.move(src_file, dest_file)
                    pbar.set_postfix(file=filename)  # Update progress bar with current file
                    pbar.update(1)  # Update progress bar by one step
                except Exception as e:
                    print(f"Error moving {filename}: {e}")


# Example usage
source_folder = '/data/ATM/temp'
destination_folder = '/mnt/Webdrive/staff-umbrella/vrlabarchive01/felix'

move_files(source_folder, destination_folder)
