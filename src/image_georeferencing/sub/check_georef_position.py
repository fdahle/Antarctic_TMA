
import image_georeferencing.sub.derive_new as dn

def check_georef_position(image_id):

    is_outlier = dn.derive_new(image_id, outlier_mode=True)

    if is_outlier:
        return False
    else:
        return True

if __name__ == "__main__":

    test = check_georef_position("CA164432V0057")
    print(test)