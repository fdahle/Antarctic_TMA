from shapely.ops import unary_union

import base.print_v as p


def merge_footprints(footprints, catch=True, verbose=False, pbar=None):
    """
    merge_footprint(footprints, catch, verbose, pbar):
    Combine a list of shapely polygons into one big polygon
    Args:
        footprints (list): A list of shapely polygons
        catch (Boolean, True): If true and something is going wrong (for example no fid points),
            the operation will continue and not crash
        verbose (Boolean, False): If true, the status of the operations are printed
        pbar (tqdm-progress-bar): If this is true, the text output will be not shown as text, but
            as a description in a tqdm-progress-bar
    Returns:
        merged (shapely-polygon): All polygons combined
    """
    p.print_v(f"Start: merge_footprints ({len(footprints)} footprints)",
              verbose=verbose, pbar=pbar)

    try:
        merged = unary_union(footprints)
    except (Exception,) as e:
        p.print_v("Merging of polygons failed", verbose, "red", pbar=pbar)
        if catch:
            return None
        else:
            raise e

    p.print_v(f"Finished: merge_footprints ({len(footprints)} footprints)",
              verbose=verbose, pbar=pbar)

    return merged
