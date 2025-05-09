"""Compresses all TIFF files in a given directory"""

# Library imports
import os
import warnings
from PIL import Image, TiffTags
from tqdm import tqdm
from typing import NoReturn

# Constants
INPUT_FLD = "/data/ATM/data_1/aerial/TMA/downloaded"
METHOD = "lzw"
QUALITY = 100


def compress_tif_files(folder_path: str,
                       compression_method: str = 'lzw',
                       quality: int = 90,
                       verbose:bool =  False) -> NoReturn:
    """
    Compresses all TIFF files in a given directory using specified compression method and quality.

    This function iterates over all TIFF files in the specified folder, applying compression based on the
    provided method ('lzw' or 'jpeg'). It uses the Python Imaging Library (PIL) to open and save the images
    with the desired compression settings. DecompressionBombWarning warnings are suppressed during the process.
    Args:
        folder_path (str): The file path of the folder containing TIFF files to be compressed.
        compression_method (str): The compression method to use ('lzw' or 'jpeg'). Defaults to 'lzw'.
        quality (int): The quality setting for JPEG compression (1-100). Higher values mean better quality but
            less compression. This parameter is ignored when using LZW compression. Defaults to 90.
    Returns:
        This function does not return anything. It compresses the files in place.
    Raises:
        Exception: Catches and ignores exceptions during image processing, logging failed attempts.
    """

    if verbose:
        print(f"Compress files in {folder_path} using "
              f"{compression_method} with quality {quality}")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)

        # Iterate through all files in the folder
        for filename in (pbar := tqdm(os.listdir(folder_path))):

            pbar.set_postfix_str(f"Compress {filename}")

            # Check if the file is a TIF file
            if filename.endswith('.tif'):
                file_path = os.path.join(folder_path, filename)
                try:

                    # Open the TIF file
                    with Image.open(file_path) as img:

                        # Get current compression type
                        current_compression = img.tag_v2.get(259)  # 259 is the compression tag
                        compression_needed = True

                        # Check if the current compression matches the desired method
                        if compression_method.lower() == 'lzw' and current_compression == 5:
                            compression_needed = False  # 5 represents LZW compression
                        elif compression_method.lower() == 'jpeg' and current_compression == 7:
                            compression_needed = False  # 7 represents JPEG compression

                        # Compress the image if needed
                        if compression_needed:
                            if compression_method.lower() == 'lzw':
                                img.save(file_path, compression='tiff_lzw')
                            elif compression_method.lower() == 'jpeg':
                                img.save(file_path, compression='jpeg', quality=quality)

                        if compression_needed:
                            pbar.set_postfix_str(f"{filename} compressed")
                        else:
                            pbar.set_postfix_str(f"{filename} already compressed")

                except (Exception,) as e:
                    pbar.set_postfix_str(f"{filename} failed")
                    print(e)
                    continue


if __name__ == "__main__":
    compress_tif_files(INPUT_FLD, METHOD, QUALITY)
