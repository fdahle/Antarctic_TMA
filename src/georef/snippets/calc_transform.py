import numpy as np
import rasterio

from affine import Affine
from osgeo import gdal, osr
from rasterio.transform import from_gcps
from sklearn.cluster import KMeans


def calc_transform(image, tps, conf=None, transform_method="rasterio",
                   gdal_order=1, no_data_value=0, epsg_code=3031,
                   simplify_points=False, n_clusters=-1):

    # Simplify GCPs if requested
    if simplify_points:

        # try to find an optimal number of tie-points
        if n_clusters == -1:
            n_clusters = __optimal_clusters(tps)

        # should be at least 50 clusters but maximum as many as the number of points
        n_clusters = np.maximum(n_clusters, 50)
        n_clusters = np.minimum(n_clusters, tps.shape[0])

        # create clusters
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
        kmeans.fit(tps[:, :2])

        # here we save later the number of clusters
        reduced_tps = []

        # we filter by centroids
        if conf is None:

            # get centroid of clusters
            centroids = kmeans.cluster_centers_

            # Find the closest original GCP for each centroid
            for centroid in centroids:
                distances = np.sqrt(((tps[:, :2] - centroid) ** 2).sum(axis=1))
                idx = np.argmin(distances)
                reduced_tps.append(tps[idx])
        else:

            # get the ids of clusters
            labels = kmeans.labels_

            # iterate every cluster
            for cluster in range(n_clusters):

                # Indices of GCPs belonging to the current cluster
                cluster_indices = np.where(labels == cluster)[0]

                # Find the index of the GCP with the highest confidence within this cluster
                idx_max_conf = cluster_indices[np.argmax(conf[cluster_indices])]

                # Append the corresponding GCP to the reduced list
                reduced_tps.append(tps[idx_max_conf])

        # save the reduced tps
        reduced_tps = np.array(reduced_tps)
        tps = reduced_tps

    # here we save the gcps
    gcps = []

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
            gcp = rasterio.control.GroundControlPoint(gcp_coords[3], gcp_coords[2],  # noqa
                                                      gcp_coords[0], gcp_coords[1])
        else:
            raise ValueError("transform method not specified")
        gcps.append(gcp)  # noqa

    # get transform matrix with gdal
    if transform_method == "gdal":

        # create the tiff file in memory
        driver = gdal.GetDriverByName("MEM")
        ds = driver.Create("", image.shape[1], image.shape[0], 1, gdal.GDT_Byte)
        for i in range(1):
            band = ds.GetRasterBand(i + 1)
            band.WriteArray(image)
            band.SetNoDataValue(no_data_value)

        # Set the GCPs for the image
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(epsg_code)
        ds.SetGCPs(gcps, srs.ExportToWkt())

        warp_options = ["-r", "near", "-et", "0"]  # , "-refine_gcps"]
        if gdal_order in [1, 2, 3]:
            warp_options.append("-order")
            warp_options.append(str(gdal_order))

        # save the new geo-referenced tiff-file
        output_ds = gdal.Warp(output_path_tiff, ds,  # noqa
                              dstSRS=srs.ExportToWkt(), options=warp_options)
        output_ds.GetRasterBand(1).SetMetadataItem("COMPRESSION", "LZW")

        # get & convert transform
        transform = output_ds.GetGeoTransform()
        transform = Affine.from_gdal(*transform)

    # get transform matrix with rasterio
    elif transform_method == "rasterio":
        transform = from_gcps(gcps)

        profile = {
            'driver': 'GTiff',
            'height': image.shape[0],
            'width': image.shape[1],
            'count': 1,
            'dtype': image.dtype,
            'crs': f'EPSG:{epsg_code}',
            'transform': transform,
            'nodata': no_data_value
        }

        transform = profile["transform"]

    else:
        raise ValueError("Transformation method not recognized")

    # Calculate Residual Errors if the transformation method was successful
    if transform:
        transformed_coords = [transform * (point[2], point[3]) for point in tps]
        residuals = [np.sqrt((x - point[0]) ** 2 + (y - point[1]) ** 2) for (x, y), point in
                     zip(transformed_coords, tps)]
    else:
        raise ValueError("Transform was not successful")

    # convert output to numpy array
    transform = np.array(transform)
    residuals = np.array(residuals)

    return transform, residuals
