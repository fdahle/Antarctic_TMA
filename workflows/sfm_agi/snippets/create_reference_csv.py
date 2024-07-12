import os

import src.sfm_agi.snippets.export_cameras as ec

_image_ids = ['CA184832V0146', 'CA184832V0147', 'CA184832V0148', 'CA184832V0149', 'CA184832V0150']
_save_fld = "/home/fdahle/Desktop/agi_test"


def create_reference_csv(image_ids, save_path):

    save_path = os.path.join(save_path, "reference.csv")

    ec.export_cameras(image_ids, save_path)


if __name__ == "__main__":
    create_reference_csv(_image_ids, _save_fld)