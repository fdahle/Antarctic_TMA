Following structure: 
1) images are detected with georef.py and saved in either
   - sat.shp and sat_processed_images.csv
   - img.shp and img_processed_images.csv
   - calc.shp and calc_processed_images.csv
2) It is possible to georeference more images by using georef_adapted.py
   - These are saved in sat.shp and adapted_processed_images.csv
3) With georef_oblique, oblique versions of the images are created and saved in
   - sat_oblique.shp
   - img_oblique.shp
   - calc_oblique.shp
4) update_georef_psql.py can be used to save the information in the database
   - There's an order: sat > img > calc
   - The exact footprint and center is saved in images_extracted
   - "Debug" information is saved in images_georef, but that only for the vertical images