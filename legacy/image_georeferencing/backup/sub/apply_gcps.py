import rasterio

from affine import Affine
from osgeo import gdal, osr
from rasterio.transform import from_gcps


def apply_gcps(output_path_tiff, image, tps,
               transform_method, gdal_order=1,
               catch=True):

    # here we save the gcps
    gcps = []
    try:
        # iterate to create the ground control points
        for i in range(tps.shape[0]):
            row = tps[i, :]

            # coords for gcps must be in float
            gcp_coords = (float(row[0]), float(row[1]), float(row[2]), float(row[3]))

            # decide on the transform methods
            if transform_method == "gdal":
                gcp = gdal.GCP(gcp_coords[0], gcp_coords[1], 0,
                               gcp_coords[2], gcp_coords[3])
            elif transform_method == "rasterio":
                gcp = rasterio.control.GroundControlPoint(gcp_coords[3], gcp_coords[2],
                                                          gcp_coords[0], gcp_coords[1])

            gcps.append(gcp)  # noqa

        if transform_method == "gdal":

            # create the tiff file in memory
            driver = gdal.GetDriverByName("MEM")
            ds = driver.Create("", image.shape[1], image.shape[0], 1, gdal.GDT_Byte)
            for i in range(1):
                band = ds.GetRasterBand(i + 1)
                band.WriteArray(image)
                band.SetNoDataValue(0)

            # Set the GCPs for the image
            srs = osr.SpatialReference()
            srs.ImportFromEPSG(3031)
            ds.SetGCPs(gcps, srs.ExportToWkt())

            warp_options = ["-r", "near", "-et", "0"]  # , "-refine_gcps"]
            if gdal_order in [1, 2, 3]:
                warp_options.append("-order")
                warp_options.append(str(gdal_order))

            # save the new geo-referenced tiff-file
            output_ds = gdal.Warp(output_path_tiff, ds,  # noqa
                                  dstSRS=srs.ExportToWkt(), options=warp_options)
            output_ds.GetRasterBand(1).SetMetadataItem("COMPRESSION", "LZW")

            transform = output_ds.GetGeoTransform()

            # convert transform
            transform = Affine.from_gdal(*transform)

            output_ds.FlushCache()
            output_ds = None  # noqa

        elif transform_method == "rasterio":
            transform = from_gcps(gcps)

            profile = {
                'driver': 'GTiff',
                'height': image.shape[0],
                'width': image.shape[1],
                'count': 1,
                'dtype': image.dtype,
                'crs': 'EPSG:3031',
                'transform': transform,
                'nodata': 0
            }

            with rasterio.open(output_path_tiff, "w", **profile) as dst:
                dst.write(image, 1)
                transform = dst.transform

        return transform

    except (Exception,) as e:
        if catch:
            return None
        else:
            raise e
