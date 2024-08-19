import os.path
import src.base.create_mask as cm
import src.export.export_tiff as et
import src.load.load_image as li
import src.base.rotate_image as ri

fld_path = "/home/fdahle/Desktop/agi_test"

img_fld_path = os.path.join(fld_path, "images")
mask_fld_path = os.path.join(fld_path, "masks")

# iterate all images in images folder
for img in os.listdir(img_fld_path):
    img_id = img.split(".")[0]

    img_path = os.path.join(img_fld_path, img)

    img = li.load_image(img_path)

    # create mask path
    mask_path = os.path.join(mask_fld_path, f"{img_id}_mask.tif")

    # create mask
    mask = cm.create_mask(img, use_database=True, image_id=img_id, uint8=True)

    if "V" in img_id:
        img = ri.rotate_image(img, 180)
        mask = ri.rotate_image(mask, 180)

    if "V" in img_id:
        et.export_tiff(img, img_path, overwrite=True)
    et.export_tiff(mask, mask_path, overwrite=True)
