# Base functions
This folder contains all base functions that are needed for all stages of the project. It follows the principle one file per function (at least most of the time, see
column Notes), so every file should only do one function (which can usually be recognize by the file name).

The following table contains an overview of all files and their functions:

| File        | Description             |  Notes   |
| ------------------------- | ------------------------------------------------------ | ------------------------------------------------------------------ |
| connect_to_db             | All interaction with the database                      | This file has multiple functions (like edit_data, get_data, etc..) |
| download_data_from_db     | Download data from a database to a sql-lite file       |                                                                    |
| download_images_from_usgs | Download images from the USGS-TMA-archive              |                                                                    |
| extract_subsets           | Extract subsets from images & train a subset extractor | This file has two different functions                              |
| get_ids_from_folder       | Get all images from a list of folders                  |                                                                    |
| load_image_from_file      | Load an image from a file into a np-array              |                                                                    |
| remove_usgs_logo          | Remove the USGS logo from newly download images        |                                                                    |
