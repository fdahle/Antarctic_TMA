from shapely.ops import unary_union

import base.print_v as p


def merge_footprints(footprints, catch=True, verbose=False, pbar=None):
    """
    merge_footprint(footprints, catch, verbose, pbar):
    Args:
        footprints:
        catch:
        verbose:
        pbar:

    Returns:

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
