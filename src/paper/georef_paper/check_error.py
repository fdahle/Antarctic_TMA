
import base.load_shape_data as lsd

path_shape_file = "/data_1/ATM/data_1/papers/paper_georef/measure_lines.shp"

shape_data = lsd.load_shape_data(path_shape_file)

shape_data['length'] = shape_data['geometry'].length

print(shape_data[['image','length', 'type']])
