import gzip
import os
import shutil
import ssl
import wget

PATH_DOWNLOAD_FLD = "/data/ATM/data_1/aerial/TMA/downloaded_child"
USGS_URL = "http://data.pgc.umn.edu/aerial/usgs/tma/photos/high"


def download_image_from_usgs(image_id, download_folder=None, url_image=None,
                             overwrite=True, catch=True, verbose=False, pbar=None):
    """
    download_image_from_usgs(ids, download_folder, add_to_db, url_images, verbose, catch):
    With this function it is possible to download an image from the TMA archive and store it in
    the download folder. The image is downloaded in a zipped format, so after downloading it
    also need to be unpacked (and sometimes the zipped file is not properly deleted).
    Args:
        image_id (String): A string with the image_id of the image.
        download_folder (String, None): The folder in which the image should be stored.
            If "None", the default download path will be used
        url_image (String, None): The url where the images can be downloaded. If "None",
            these will be downloaded from the University of Minnesota
        overwrite (Boolean): If true and the image is already existing we're not downloading again
        catch (Boolean): If true, we catch every error that is happening and return instead None
        verbose (Boolean): If true, we print information that is happening during execution of the function
        pbar (Tqdm-progressbar): If this is not None, the printing will be added to a tqdm-progress bar

    Returns:
        img_path (String): The path to which the images have been downloaded.
    """

    print(f"Download {image_id} from USGS")

    # specify the download folder
    if download_folder is None:
        download_folder = PATH_DOWNLOAD_FLD

    # from here we can download the images
    if url_image is None:
        url_image = USGS_URL

    # important as otherwise the download fails sometimes
    ssl._create_default_https_context = ssl._create_unverified_context  # noqa

    # get the image params
    tma = image_id[0:6]
    direction = image_id[6:9]
    roll = image_id[9:13]

    # define the url and download paths
    url = url_image + "/" + tma + "/" + tma + direction + "/" + tma + direction + roll + ".tif.gz"
    gz_path = download_folder + "/" + image_id + ".gz"
    img_path = download_folder + "/" + image_id + ".tif"

    # check if we already have this file and if we want to overwrite it
    if overwrite is False and os.path.isfile(img_path):
        print(f"{image_id} is already downloaded")
        return

    try:
        # download file
        wget.download(url, out=gz_path)

        # unpack file
        with gzip.open(gz_path) as f_in, open(img_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

        # remove the gz file
        os.remove(gz_path)

        print(f"{image_id} downloaded successfully")

        return True

    except (Exception,) as e:
        if catch:
            print(f"{image_id} downloaded failed")
            return False
        else:
            raise e

if __name__ == "__main__":

    image_ids = [
        "CA000400V0063", "CA000400V0063", "CA000400V0064", "CA000400V0065",
        "CA000400V0066", "CA000400V0067", "CA000400V0068", "CA000400V0069",
        "CA000400V0070", "CA000400V0071", "CA000400V0072", "CA000400V0073",
        "CA000400V0074", "CA000400V0075", "CA000400V0076", "CA000400V0077",
        "CA000400V0078", "CA000400V0079", "CA000400V0080", "CA000400V0081",
        "CA000400V0082", "CA000400V0083", "CA000400V0084", "CA000400V0085",
        "CA000400V0086", "CA000400V0087", "CA000400V0088", "CA000400V0089",
        "CA000400V0090", "CA000400V0091", "CA000400V0117", "CA000400V0118",
        "CA000400V0119", "CA000400V0120", "CA000400V0121", "CA000400V0122",
        "CA000400V0123", "CA000400V0124", "CA000400V0125", "CA000400V0126",
        "CA000400V0127", "CA000400V0128", "CA000400V0129", "CA000400V0130",
        "CA000400V0131", "CA000400V0132", "CA000400V0133", "CA000400V0134",
        "CA000400V0135", "CA000400V0136", "CA000400V0137", "CA000400V0138",
        "CA000400V0139", "CA000400V0140", "CA000400V0141", "CA000400V0142",
        "CA000400V0143", "CA000400V0144", "CA000400V0145", "CA000400V0146",
        "CA000400V0153", "CA000400V0154", "CA000400V0155", "CA000400V0156",
        "CA000400V0157", "CA000400V0158", "CA000400V0159", "CA000400V0160",
        "CA000400V0161", "CA000400V0162", "CA000400V0163", "CA000400V0164",
        "CA000400V0165", "CA000400V0166", "CA000400V0167", "CA000400V0168",
        "CA000400V0169", "CA000400V0170", "CA000400V0171", "CA000400V0172",
        "CA000400V0173", "CA000400V0174", "CA000400V0175", "CA000400V0176",
        "CA000400V0177", "CA000400V0197", "CA000400V0198", "CA000400V0199",
        "CA000400V0200", "CA000400V0201", "CA000400V0202", "CA000400V0203",
        "CA000400V0204", "CA000400V0205", "CA000400V0206", "CA000400V0207",
        "CA000400V0208", "CA000400V0209", "CA000400V0210", "CA000400V0211",
        "CA000400V0212", "CA000400V0213", "CA000400V0214", "CA000400V0215",
        "CA000400V0216", "CA000400V0217", "CA000400V0218", "CA000400V0219",
        "CA000400V0220", "CA000400V0227", "CA000400V0228", "CA000400V0229",
        "CA000400V0230", "CA000400V0231", "CA000400V0232", "CA000400V0233",
        "CA000400V0234", "CA000400V0235", "CA000400V0236", "CA000400V0237",
        "CA000400V0238", "CA000400V0239", "CA000400V0240", "CA000400V0241",
        "CA000400V0242", "CA000400V0243", "CA000400V0244", "CA000400V0245",
        "CA000400V0246", "CA000400V0247", "CA000400V0248", "CA000400V0249",
        "CA000500V0017", "CA000500V0018", "CA000500V0019", "CA000500V0020",
        "CA000500V0021", "CA000500V0022", "CA000500V0023", "CA000500V0024",
        "CA000500V0025", "CA000500V0026", "CA000500V0027", "CA000500V0028",
        "CA000500V0029", "CA000500V0030", "CA000500V0031", "CA000500V0032",
        "CA000500V0033", "CA000500V0039", "CA000500V0040", "CA000500V0041",
        "CA000500V0042", "CA000500V0043", "CA000500V0044", "CA000500V0045",
        "CA000500V0046", "CA000500V0047", "CA000500V0048", "CA000500V0049",
        "CA000500V0050", "CA000500V0051", "CA000500V0052", "CA000500V0053",
        "CA000500V0054", "CA000500V0055", "CA000500V0056", "CA000900V0078",
        "CA000900V0079", "CA000900V0080", "CA000900V0081", "CA000900V0082",
        "CA000900V0083", "CA000900V0084", "CA000900V0085", "CA000900V0086",
        "CA000900V0087", "CA000900V0088", "CA000900V0089", "CA000900V0090",
        "CA000900V0091", "CA000900V0092", "CA000900V0093", "CA000900V0094",
        "CA000900V0095", "CA000900V0096", "CA000900V0097", "CA000900V0098",
        "CA000900V0099", "CA000900V0100", "CA000900V0101", "CA000900V0102",
        "CA000900V0103", "CA000900V0104", "CA000900V0105", "CA000900V0106",
        "CA000900V0107", "CA000900V0108", "CA000900V0109", "CA000900V0140",
        "CA000900V0141", "CA000900V0142", "CA000900V0143", "CA000900V0144",
        "CA000900V0145", "CA000900V0146", "CA000900V0147", "CA000900V0148",
        "CA000900V0149", "CA000900V0150", "CA000900V0151", "CA000900V0152",
        "CA000900V0153", "CA000900V0154", "CA000900V0155", "CA000900V0156",
        "CA000900V0157", "CA000900V0158", "CA000900V0159", "CA000900V0160",
        "CA000900V0161", "CA000900V0162", "CA000900V0163", "CA000900V0164",
        "CA000900V0165", "CA000900V0166", "CA000900V0185", "CA000900V0186",
        "CA000900V0187", "CA000900V0188", "CA000900V0189", "CA000900V0190",
        "CA000900V0191", "CA000900V0192", "CA000900V0193", "CA000900V0194",
        "CA000900V0195", "CA000900V0196", "CA000900V0197", "CA000900V0198",
        "CA000900V0199", "CA000900V0200", "CA000900V0201", "CA000900V0202",
        "CA000900V0203", "CA000900V0204", "CA000900V0205", "CA000900V0206",
        "CA000900V0207", "CA000900V0208", "CA000900V0209", "CA001000V0007",
        "CA001000V0008", "CA001000V0009", "CA001000V0010", "CA001000V0011",
        "CA001000V0012", "CA001000V0013", "CA001000V0014", "CA001000V0015",
        "CA001000V0016", "CA001000V0017", "CA001000V0018", "CA001000V0019",
        "CA001000V0020", "CA001000V0021", "CA001000V0022", "CA001000V0023",
        "CA001000V0024", "CA001000V0025", "CA001000V0026", "CA001000V0027",
        "CA001000V0047", "CA001000V0048", "CA001000V0049", "CA001000V0050",
        "CA001000V0051", "CA001000V0052", "CA001000V0053", "CA001000V0054",
        "CA001000V0055", "CA001000V0056", "CA001000V0057", "CA001000V0058",
        "CA001000V0059", "CA001000V0060", "CA001000V0061", "CA001000V0062",
        "CA001000V0063", "CA001000V0064", "CA001000V0065", "CA001000V0066",
        "CA001000V0073", "CA001000V0074", "CA001000V0075", "CA001000V0076",
        "CA001000V0077", "CA001000V0078", "CA001000V0079", "CA001000V0080",
        "CA001000V0081", "CA001000V0082", "CA001000V0083", "CA001000V0084",
        "CA001000V0085", "CA001000V0086", "CA001000V0087", "CA001000V0088",
        "CA001000V0089", "CA001000V0090", "CA001000V0091", "CA001000V0092",
        "CA001000V0093", "CA001000V0094", "CA001000V0095", "CA001000V0108",
        "CA001000V0109", "CA001000V0110", "CA001000V0112", "CA001000V0113",
        "CA001000V0114", "CA001000V0115", "CA001000V0116", "CA001000V0117",
        "CA001000V0118", "CA001000V0119", "CA001000V0120", "CA001000V0121"
    ]

    from tqdm import tqdm
    for image_id in tqdm(image_ids):
        download_image_from_usgs(image_id)
