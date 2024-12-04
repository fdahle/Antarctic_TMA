# Required for streamlit
import sys
from pathlib import Path
src_path = (Path(__file__).parent.parent / 'src').resolve()
print(f"Adding {src_path} to sys.path")
if str(src_path) not in sys.path:
    sys.path.append(str(src_path))

# Library imports
import math
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import streamlit as st


# Constants
PATH_MAIN_FOLDER = "/data/ATM/data_1/aerial/TMA"  # images in this folder and sub-folders will be checked


def verify_images():

    base_path = Path(PATH_MAIN_FOLDER)

    subfolders_files = {subfolder.name: {file.name for file in subfolder.glob('*.tif')}
                        for subfolder in base_path.iterdir() if subfolder.is_dir()}

    # Create a union of all files across subfolders to identify the complete set of expected files
    all_files_union = set().union(*subfolders_files.values())

    # create return dict and fill already with total number of images
    return_dict={"nr_images": len(all_files_union)}

    # Identify missing images in each subfolder
    missing_images = {subfolder: all_files_union - files for subfolder, files in subfolders_files.items()}

    return_dict["missing_images"] = missing_images

    return return_dict

def verify_image_sizes():

    # get the image sizes from database
    pass

    #conn = ctd.

def plot_results():

    # set the title
    st.title("Files Quality Control")

    st.header("Missing Images")

    # get the required information
    return_dict= verify_images()

    # Determine grid size
    n = len(return_dict["missing_images"])
    cols_in_grid = 5  # Number of columns in the grid
    rows_in_grid = math.ceil(n / cols_in_grid)

    # Create the figure and axes
    fig, axs = plt.subplots(rows_in_grid, cols_in_grid, figsize=(15, rows_in_grid * 3))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)  # Adjust space between plots

    # Flatten the array of axes for easy iteration
    axs = axs.ravel()

    i = 0
    for key, val in return_dict["missing_images"].items():
        fill_rate = 100 - (len(val) / return_dict["nr_images"] * 100)

        # Calculate color based on fill_rate
        color = mcolors.to_hex([1 - fill_rate / 100, fill_rate / 100, 0])

        axs[i].bar(key, fill_rate, color=color)
        axs[i].set_ylim(0, 100)
        axs[i].set_title(key)
        axs[i].set_ylabel("% Filled")
        axs[i].set_xticklabels([round(fill_rate, 1)])

        axs[i].annotate(f'{fill_rate}%',
                        xy=(0.5, fill_rate),  # Adjust x position to center
                        xytext=(0, 10),  # 10 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

        i += 1

    # Hide any unused axes if the number of plots is not a perfect fill of the grid
    for ax in axs[n:]:
        ax.set_visible(False)

    st.pyplot(fig)


if __name__ == "__main__":


    plot_results()
