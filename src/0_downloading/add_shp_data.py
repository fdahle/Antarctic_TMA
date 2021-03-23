"""
This algorithm does the following things:
- iterate all image files in table 'images'
- checks if these files have the information from the shp 'TMA_Photocenters'
- if not add this information to the table
"""
import sys #in order to import other python files
from datetime import date #to work with dates
import geopandas as gpd #necessary to read the shp files

#add for import
sys.path.append('../misc')

from database_connections import *

verbose = True #specify if stuff should be printed
shp_file_path = "../../data/shapes/TMA_Photocenters/TMA_pts_20100927.shp"

def add_shp_data(shp_file):

    #create connection to db
    conn = Connector()

    #specify params for table select
    tables = ["images"]
    fields = [["photo_id"]]
    filters = [
        {"point_x": "NULL"},
        {"point_y": "NULL"},
        {"quad_name": "NULL"},
        {"location": "NULL"},
        {"flying_direction": "NULL"},
        {"id_cam": "NULL"},
        {"altitude": "NULL"},
        {"azimuth": "NULL"},
        {"date_of_recording":"NULL"}
    ]

    #get data from table
    tableData = conn.get_data(tables, fields, filters)

    #get data from shp file
    shapeData = gpd.read_file(shp_file)

    for idx, tableRow in tableData.iterrows():

        #get the unique id to extract from shp
        photo_id = tableRow["photo_id"]
        entity_id = photo_id[0:6] + photo_id[-4:]

        #extract row from shp
        row = shapeData[shapeData['ENTITY_ID'] == entity_id]

        table="images"

        #which entry should be updated
        id = {
            "photo_id": photo_id
        }

        #convert the date_of_recording
        date_of_rec = row["Date_"].values[0]
        if len(date_of_rec.split("/")) != 3:
            date_formatted = "NULLDATE()"
        else:
            day, month, year = date_of_rec.split("/")
            if int(day) < 10:
                day = "0" + day
            if int(month) < 10:
                month = "0" + month
            date_formatted = day + "/" + month + "/" + year

        #the new data that will be saved
        data = {
            "point_x": row["POINT_X"].values[0],
            "point_y": row["POINT_Y"].values[0],
            "quad_name": row["QuadName"].values[0],
            "location": row["Location"].values[0],
            "flying_direction": row["Direction"].values[0],
            "id_cam": row["v_cam"].values[0],
            "altitude": row["Altitude"].values[0],
            "azimuth": row["Azimuth_dd"].values[0],
            "date_of_recording": date_formatted
        }

        #update
        conn.edit_data(table, id, data)

if __name__ == "__main__":
    add_shp_data(shp_file_path)
