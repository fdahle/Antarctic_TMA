from shapely.geometry import Polygon

def get_overlapping_images(image_ids, footprints):

    overlap_dict = {}

    # match every image with every image
    for i in range(len(image_ids)):

        # Ensure each pair is only compared once
        for j in range(i + 1, len(image_ids)):

            # get data of image 1
            image_id1 = image_ids[i]
            footprint1 = footprints[i]

            # get data of image 2
            image_id2 = image_ids[j]
            footprint2 = footprints[j]

            # check if images are overlapping
            if footprint1.intersects(footprint2):

                # create entry for image1
                if image_id1 not in overlap_dict:
                    overlap_dict[image_id1] = []
                overlap_dict[image_id1].append(image_id2)

                # create entry for image2
                if image_id2 not in overlap_dict:
                    overlap_dict[image_id2] = []
                overlap_dict[image_id2].append(image_id1)