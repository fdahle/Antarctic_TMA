import os
import warnings

from tqdm import tqdm
from PIL import Image

INPUT_FLD = "/data_1/ATM/data_1/playground/georef4/tiffs/calc"
METHOD = "jpeg"
QUALITY = 90


def compress_tif_files(folder_path, compression_method='lzw', quality=90):

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
                        # Determine the compression method
                        if compression_method.lower() == 'lzw':
                            # Use LZW compression
                            img.save(file_path, compression='tiff_lzw')

                        elif compression_method.lower() == 'jpeg':
                            # Use JPEG compression within the TIFF format
                            # This option can introduce loss but retains the TIFF format
                            img.save(file_path, compression='jpeg', quality=quality)  # Adjust quality as needed

                        pbar.set_postfix_str(f"{filename} compressed")

                except (Exception,) as _:
                    pbar.set_postfix_str(f"{filename} failed")


if __name__ == "__main__":
    compress_tif_files(INPUT_FLD, METHOD, QUALITY)
