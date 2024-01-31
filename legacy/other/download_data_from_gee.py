import ee
import os
import json
import shutil
import time

from datetime import datetime
from oauth2client.service_account import ServiceAccountCredentials
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

import base.print_v as p


def download_data_from_gee(polygon, output_folder=None,
                           key_path=None, delete_files_on_gdrive=True,
                           satellite_type="S2", epsg_code=3031,
                           month=0,
                           overwrite=False, catch=True, verbose=False, pbar=None):
    """
    download_data_from_gee(bounds, destination_folder, satellite_type, delete_files, epsg_code, verbose):
    This function can automatically download data from the Google Earth Engine of a certain area of interest (aoi).
    Currently, this data is creating cloud-free composites of satellite images over multiple years and download them.
    Note that in order to work a json file with keys (for access to Google) is required.
    Args:
        polygon (shapely-polygon): A shapely polygon describing the area we want to download
        output_folder (String, None): The folder in which the data should be downloaded to.
        key_path (String, None): The folder in which the keys for authenticating with GEE are
        satellite_type (String, "S2"): From which satellite the data is used. Currently, only Sentinel 2 is supported.
        delete_files_on_gdrive (Boolean, True): The files must be temporarily stored on a Google Drive (and then
        downloaded from there). If true, the files on the G-drive will be deleted after download. Recommended to set
        to "True", as otherwise the Drive storage gets full.
        epsg_code (Int, 3031): The epsg-code in which want to have the data.
        overwrite (Boolean, False): If true, the downloaded date will be overwritten
        catch (Boolean, True): If true and something is going wrong, the operation will continue
            and not crash
        verbose (Boolean, False): If true, the status of the operations are printed.
        pbar (tqdm-progress-bar, None): If this is true, the text output will be not shown as text, but
            as a description in a tqdm-progress-bar

    Returns:
        return_path (String): The path where the downloaded file can be found
    """

    assert 0 <= month <= 12

    # load the json to get default values
    json_folder = os.path.dirname(os.path.realpath(__file__))
    with open(json_folder + "/params.json") as j_file:
        json_data = json.load(j_file)

    # specify default value for output folder
    if output_folder is None:
        output_folder = json_data["path_folder_gee"]

    month_strings = [
        "0_complete", "1_jan", "2_feb", "3_mar", "4_apr", "5_may", "6_jun",
        "7_jul", "8_aug", "9_sep", "10_oct", "11_nov", "12_dec"
    ]

    output_folder = output_folder + "/" + month_strings[month]

    # specify default value for the key
    if key_path is None:
        key_path = json_data["path_folder_gee_keys"]

    if satellite_type == "S2":
        collection_link = "COPERNICUS/S2"
    else:
        collection_link = None  # noqa
        raise NotImplementedError

    # define size of polygon & create an EE-specific polygon
    min_x = polygon.bounds[0]
    max_x = polygon.bounds[2]
    min_y = polygon.bounds[1]
    max_y = polygon.bounds[3]
    polygon_ee = [[[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y], [min_x, min_y]]]

    # set file name
    output_name = f"{round(min_x)}_{round(max_x)}_{round(min_y)}_{round(max_y)}"

    # define the final destination path
    destination_path = output_folder + "/" + output_name + ".tif"

    # check if the file is already existing
    if os.path.isfile(destination_path) and overwrite is False:
        p.print_v(f"{output_name + '.tif'} is already existing", verbose, "green", pbar=pbar)
        return True

    try:

        # get the service account
        with open(key_path + '/service_worker_mail.txt', 'r') as file:
            service_account_file = file.read().replace('\n', '')

        # get the credentials
        credentials = ee.ServiceAccountCredentials(service_account_file, key_path + "/gee_secret.json")

        # authenticate and initialize Google Earth Engine
        ee.Initialize(credentials)

        # some parameters of the Sentinel2-collection
        aoi = ee.Geometry.Polygon(polygon_ee, "EPSG:3031", evenOdd=False)
        start_date = '2016-01-01'
        end_date = '2022-12-31'
        cloud_filter = 50  # Maximum image cloud cover percent allowed in image collection

        # mask based on clouds
        def mask_s2_clouds(image):
            qa = image.select('QA60')

            # left shift operator
            cloud_bit_mask = 1 << 10
            cirrus_bit_mask = 1 << 11

            # create mask
            cloud_mask = qa.bitwiseAnd(cloud_bit_mask).eq(0)
            cirrus_mask = qa.bitwiseAnd(cirrus_bit_mask).eq(0)

            # combine masks
            mask = cloud_mask.And(cirrus_mask)

            # Return the masked and scaled data, without the QA bands.
            return image.updateMask(mask).divide(10000).multiply(255)

        # build a collection and filer based on AOI, date and cloud cover
        if month > 0:
            collection = (ee.ImageCollection(collection_link)
                          .filterBounds(aoi)
                          .filterDate(start_date, end_date)
                          .filter(ee.Filter.calendarRange(month, month, 'month'))
                          .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', cloud_filter))).map(mask_s2_clouds)
        else:
            collection = (ee.ImageCollection(collection_link)
                          .filterBounds(aoi)
                          .filterDate(start_date, end_date)
                          .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', cloud_filter))).map(mask_s2_clouds)

        # keep only certain bands
        collection = collection.select(['B2', 'B3', 'B4'])

        # Map the function over one year of data and take the median.
        composite = collection.median()

        # clip to region of interest
        composite = composite.clip(aoi)

        # cast the image to 8 bit (to have values from 0 to 255
        composite = composite.toUint8()

        # options for export
        export_config = {
            'scale': 10,
            'region': aoi,
            'crs': "EPSG:" + str(epsg_code),
            'maxPixels': 26549123161
        }

        # export to gdrive (the only way to store files over 32MB)
        task = ee.batch.Export.image.toDrive(composite, output_name, **export_config)
        task.start()

        print("Start exporting image to google drive")

        # get start time of task
        start = datetime.now()

        # Monitor the task to wait until the data is downloaded.
        while task.status()['state'] in ['READY', 'RUNNING']:  # still running

            # wait for x seconds
            time.sleep(30)

            # get time difference
            now = datetime.now()
            diff = now - start
            days, seconds = diff.days, diff.seconds
            hours = days * 24 + seconds // 3600
            minutes = (seconds % 3600) // 60
            seconds = seconds % 60

            if verbose:
                print(f"Task is still running ({hours}:{minutes}:{seconds})")

        # if the exporting is finished
        else:
            print(task.status())

            if task.status()['state'] == 'FAILED':
                print("The export failed. Please try again")
                print(task.status()["error_message"])
                return
            else:
                print("File exported successfully")

        # set scope to google drive
        scopes = ['https://www.googleapis.com/auth/drive']

        # authenticate to Drive
        g_auth = GoogleAuth()
        g_auth.credentials = ServiceAccountCredentials.from_json_keyfile_name(key_path + "/gee_secret.json",
                                                                              scopes=scopes)
        drive = GoogleDrive(g_auth)

        # get list of files in the folder for the satellite images
        file_list = drive.ListFile({'q': "'root' in parents and trashed=false"}).GetList()
        for file1 in file_list:

            # get filename
            file_name = file1['title']

            # download file into working dir
            file1.GetContentFile(file_name, mimetype="image/tiff")

            # delete file on Gdrive
            if delete_files_on_gdrive:
                file1.Delete()

            # get current location of file
            source_path = os.getcwd() + "/" + file_name

            # get new path
            destination_path = output_folder + "/" + file_name

            # copy file to final destination folder
            shutil.move(source_path, destination_path)

        if verbose:
            print(f"Finished downloading the file. File is located at {os.path.abspath(destination_path)}")

    except (Exception,) as e:
        if catch:
            p.print_v("Something went wrong downloading the file.", verbose, "red", pbar=pbar)
        else:
            raise e


if __name__ == "__main__":

    import base.load_shape_data as lsd

    path_rema_shape = "/data_1/ATM/data_1/shapefiles/REMA/REMA_Tile_Index_Rel1_1.shp"

    # we want to download the tiles based on the mosaic of the rema data
    rema_shape_data = lsd.load_shape_data(path_rema_shape)

    tiles = ["48_06", "47_06", "47_05", "47_04", "46_07", "46_06", "46_05", "46_04",
             "45_07", "45_06", "45_05", "45_04", "44_08", "44_07", "44_06", "44_05",
             "44_04", "43_09", "43_08", "43_07", "43_06", "43_05", "42_11", "42_10",
             "42_09", "42_08", "42_07", "42_06", "42_05", "41_13", "41_12", "41_11",
             "41_10", "41_09", "41_08", "41_07", "41_06", "40_15", "40_14", "40_13",
             "40_12", "40_11", "40_10", "40_09", "40_08", "40_07", "39_17", "39_16",
             "39_15", "39_14", "39_13", "39_12", "39_11", "39_10", "39_09", "39_08",
             "39_07", "38_17", "38_16", "38_15", "38_14", "38_13", "38_12", "38_11",
             "38_10", "38_09", "38_08", "37_17", "37_16", "37_15", "37_14", "37_13",
             "37_12", "37_11", "37_10", "37_09", "37_08", "36_17", "36_16", "36_15",
             "36_14", "36_13", "36_12", "36_11", "36_10", "36_09", "35_17", "35_16",
             "35_15", "35_14", "35_13", "35_12", "35_11", "34_17", "34_16", "34_15",
             "34_14", "34_13", "34_12", "34_11", "23_15", "24_14", "25_15", "23_14",
             "24_15"]

    tiles = ["40_12"]

    #for month in [1,2,3,4,5,6,7,8,9,10,11,12]:
    for month in [0]:

        # iterate rows to download the equivalent satellite images
        for index, row in rema_shape_data.iterrows():

            # get the name (based on the tile
            tile = row["tile"]

            if tile not in tiles:
                continue

            # get the polygon
            poly = row["geometry"]

            # download the data
            download_data_from_gee(poly, month=month, catch=True, verbose=True)

