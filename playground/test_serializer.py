params = {
    'project_name': 'thunder_glacier',
    'gcp_accuracy': (30, 30, 10),
    'azimuth': None,
    'absolute_bounds': (-2500831.6606667596, 1227342.067977426, -2469061.683428534, 1250732.3420087446),
    'epsg_code': 3031, 'overwrite': False, 'resume': True, 'fixed_focal_length': True,
    'use_rotations_only_for_tps': True, 'pixel_size': 0.025, 'resolution_rel': 0.001, 'resolution_abs': 2,
    'matching_method': 'combined', 'min_overlap': 0.25, 'step_range': 2, 'custom_matching': True, 'min_tps': 15,
    'max_tps': 10000, 'min_tp_confidence': 0.9, 'tp_type': float, 'tp_tolerance': 0.5, 'custom_markers': False,
    'zoom_level_dem': 10, 'use_gcp_mask': True,
    'mask_type': ['rock', 'confidence', 'slope'],
    'rock_mask_type': 'REMA',
    'mask_resolution': 10, 'min_gcp_required': 5, 'min_gcp_confidence': 0.9, 'gcp_accuracy_px': 5,
    'max_gcp_error_px': 0.5, 'max_gcp_error_m': 50, 'max_slope_begin': 20, 'max_slope_finish': 40, 'mask_buffer': 10,
    'no_data_value': -9999, 'interpolate': True}

def custom_serializer(obj):
    """
    Custom serializer for JSON encoding.
    Converts non-serializable objects (e.g., sets, tuples) to serializable formats.
    """
    if isinstance(obj, set):  # Convert sets to lists
        return list(obj)
    elif hasattr(obj, '__dict__'):  # Convert objects with `__dict__` attribute to dicts
        return obj.__dict__
    elif isinstance(obj, type):  # Convert type objects to their string representation
        return obj.__name__

# Define a temporary path
import os
output_path_params = os.path.join("/tmp", "project_params.json")

# save the parameters of the project
import json
with open(output_path_params, 'w') as f:
    json.dump(params, f, indent=4, default=custom_serializer)  # noqa
