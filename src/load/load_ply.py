"""load ply data (usually data that is not a point cloud)"""

import numpy as np
from plyfile import PlyData

def load_ply(ply_path):
    ply_data = PlyData.read(ply_path)

    # Access the vertex data
    vertex_data = ply_data['vertex'].data

    # Dynamically extract all available properties in the vertex data
    properties = vertex_data.dtype.names
    extracted_data = [vertex_data[prop] for prop in properties]

    # Stack the extracted data column-wise
    result_array = np.column_stack(extracted_data)

    return result_array