import numpy as np
import rasterio

from affine import Affine
from osgeo import gdal, osr
from rasterio.transform import from_gcps
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import base.print_v as p

import display.display_images as di

debug_show_simplified_tps = True

debug_print_error = False


def apply_gcps(output_path_tiff, image, tps,
               transform_method, gdal_order=1,
               simplify_points=False, n_clusters=-1, conf=None,
               return_error=False, save_image=True,
               catch=True):
    """
    apply_gcps(output_path_tiff, image, tps, transform_method, gdal_order, save_image, catch)
    This function applies tie-points to a not geo-referenced image in order to geo-reference it.
    In order to achieve this, the image must be saved as a tiff-file somewhere (no in-memory
    geo-referencing)
    Args:
        output_path_tiff (String): The location where the geo-referenced image should be saved
        image (np-array): The image (as a np-array) that should be geo-referenced.
        tps (np-array): Np-array in the shape (Y,4), with the first two columns being x_abs, y_abs and
            the last two columns being x_rel, y_rel
        transform_method (String): The image can be geo-referenced using 'gdal' or 'rasterio'
        gdal_order (int, 1): the order of geo-referencing, only available for gdal. As higher the number,
            as more deformed the images can be
        simplify_points (Bool, False): If true, the number of gcp is reduced if there's too many tie-points. This makes
            the geo-referencing more accurate. For how is simplified see the equivalent place in the code
        n_clusters (int, -1): The number of clusters we want to have, if the value is -1, an optimal number of clusters
            is auto-detected (This number is then equivalent to the number of cps; this number should be higher than 50)
        conf (list, None): A list with the confidence values of each gcps, can be used in simplify_points
        return_error (Bool, False): If true, the residual error of geo-referencing is returned
        save_image (Boolean, true): Should a geo-referenced image be saved at the output_path
        catch (Boolean, True): If true and something is going wrong, the operation will continue and not crash.
            In this case None is returned
    Returns:
        transform (np-array): A 3x2 transformation matrix describing the geo-referencing
    """

    # Reduce number of GCPs using spatial clustering if n_clusters is specified
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

        if debug_show_simplified_tps:
            di.display_images([image, image], points=[tps[:, 2:4], reduced_tps[:, 2:4]])

    try:

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

            # get & convert transform
            transform = output_ds.GetGeoTransform()
            transform = Affine.from_gdal(*transform)

            if save_image:
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

            if save_image:
                with rasterio.open(output_path_tiff, "w", **profile) as dst:
                    dst.write(image, 1)
                    transform = dst.transform
            else:
                transform = profile["transform"]

        # Calculate Residual Errors if the transformation method was successful
        if transform:
            transformed_coords = [transform * (point[2], point[3]) for point in tps]
            errors = [np.sqrt((x - point[0]) ** 2 + (y - point[1]) ** 2) for (x, y), point in
                      zip(transformed_coords, tps)]
            residual_error = np.mean(errors)

            if debug_print_error:
                p.print_v(f"Average Residual Error: {residual_error:.2f} units")
        else:
            raise ValueError("Transform was not successful")

        if return_error:
            return transform, residual_error
        else:
            return transform

    except (Exception,) as e:
        if catch:
            if return_error:
                return None, None
            else:
                return None
        else:
            raise e


def __optimal_clusters(tps, max_k=50, top_n_changes=3):
    distortions = []
    sil_scores = []

    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=0).fit(tps[:, :2])
        distortions.append(kmeans.inertia_)  # noqa
        sil_scores.append(silhouette_score(tps[:, :2], kmeans.labels_))  # noqa

    # Elbow method
    changes = [distortions[i] - distortions[i + 1] for i in range(len(distortions) - 1)]
    largest_changes_indices = np.argsort(changes)[-top_n_changes:]
    elbow_point = largest_changes_indices[-1] + 2  # You may adjust how you pick from the top changes

    best_silhouette = np.argmax(sil_scores) + 2

    return max(elbow_point, best_silhouette)
