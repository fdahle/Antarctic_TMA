import getpass
import os
import shutil

from datetime import datetime
from PIL import Image

settings = {
    "image_size": (250, 250)
}

template_path = "/home/fdahle/Documents/GitHub/Antarctic_TMA/src/sfm/html/status_template.html"


def update_html(project_folder):
    # copy the template to the project folder
    shutil.copyfile(template_path, os.path.join(project_folder, "status.html"))

    # set the paths
    html_path = os.path.join(project_folder, "status.html")
    images_orig_folder = os.path.join(project_folder, "images_orig")
    thumbnails_folder = os.path.join(project_folder, "thumbnails")
    homol_folder = os.path.join(project_folder, "Homol")

    # create thumbnails folder if it does not exist
    if not os.path.exists(thumbnails_folder):
        os.makedirs(thumbnails_folder)

    # get html content
    with open(html_path, "r") as file:
        html_content = file.read()

    # basic project infos
    html_content = html_content.replace("{PROJECT_NAME}", os.path.basename(project_folder))
    html_content = html_content.replace("{PROJECT_AUTHOR}", getpass.getuser())
    current_datetime = datetime.now()
    html_content = html_content.replace("{PROJECT_CREATION_DATE}", current_datetime.strftime("%m/%d/%Y, %H:%M:%S"))

    # set the html for images
    image_files = sorted(os.listdir(images_orig_folder), key=lambda x: x.split("_")[0].split(".")[0])
    images_html = '<div class="images-container">\n'
    for image_name in image_files:

        # skip non-tif files
        if image_name.endswith(".tif") is False:
            continue

        # get the image id
        image_id = image_name.split("_")[0].split(".")[0]

        # set the paths
        original_path = os.path.join(images_orig_folder, image_name)
        thumbnail_name = os.path.splitext(image_name)[0] + ".png"
        thumbnail_path = os.path.join(thumbnails_folder, thumbnail_name)

        # create or update the thumbnail
        _create_or_update_thumbnail(original_path, thumbnail_path, settings["image_size"])

        # generate the html for the image
        relative_thumbnail_path = os.path.join("thumbnails", thumbnail_name)
        images_html += f'''
            <div class="image-block">
                <img src="{relative_thumbnail_path}" alt="{image_name}">
                <br><strong>{image_id}</strong>
                {_generate_presence_html(image_id, project_folder)}
            </div>
        '''

    # replace the placeholder with the actual images
    images_html += '</div>\n'
    html_content = html_content.replace("{PROJECT_IMAGES}", images_html)

    # get the steps html
    html_content = html_content.replace("{MICMAC_STEPS}", _update_steps_html(project_folder))

    # get tie-points html
    html_content = html_content.replace("{TIE_POINTS}", _generate_tie_points_html(homol_folder))

    # save the updated html
    with open(html_path, 'w') as file:
        file.write(html_content)


def _check_image_presence(image_id, folder):
    for fname in os.listdir(folder):
        if fname.startswith(image_id):
            return True
    return False


def _create_or_update_thumbnail(original_path, thumbnail_path, size):
    if os.path.exists(thumbnail_path):
        with Image.open(thumbnail_path) as img:
            if img.size == size:
                return

    with Image.open(original_path) as img:
        img.thumbnail(size)
        img.save(thumbnail_path, "PNG")


def _generate_presence_html(image_id, project_folder):
    directories = {
        "original images": "images_orig",
        "resampled images": "images",
        "original masks": "masks_orig",
        "resampled masks": "masks"
    }

    html_content = '<ul class="presence-list">'
    for label, dir_name in directories.items():
        present = _check_image_presence(image_id, os.path.join(project_folder, dir_name))
        class_name = "check" if present else "cross"
        symbol = "✔️" if present else "❌"
        html_content += f'<li class="{class_name}">{label}: {symbol}</li>'
    html_content += '</ul>'
    return html_content

def _update_steps_html(project_folder):
    steps = ["AperiCloud", "GCPConvert", "Malt", "Nuage2Ply", "ReSampFid", "Schnaps", "Tarama", "Tapas", "Tawny"]
    stats_folder = os.path.join(project_folder, "stats")

    html_content = ''
    for step in steps:
        json_file_path = os.path.join(stats_folder, f"{step}_stats.json")

        # Check if the JSON file exists
        if os.path.isfile(json_file_path):
            status_message = "existing"
            status_html = f"{step} {status_message}"
        else:
            status_message = "not existing"
            # Apply gray color for "not existing" message
            status_html = f'<span style="color: gray;">{step} {status_message}</span>'

        # Append the step's HTML block to the overall content
        html_content += f'''
        <div>
            <h3>{step}</h3>
            <div>{status_html}</div>
        </div>\n'''

    return html_content

def _generate_tie_points_html(homol_folder):
    connections = _parse_homol_directory(homol_folder)

    # Create a set of all unique image IDs
    image_ids = set()
    for connection in connections:
        image_ids.add(connection['image_id'])
        image_ids.add(connection['connected_to'])

    # Initialize a matrix to hold tie-points data
    tie_points_matrix = {image_id: {other_id: 0 for other_id in image_ids} for image_id in image_ids}

    # Populate the matrix with actual tie-points data
    for connection in connections:
        tie_points_matrix[connection['image_id']][connection['connected_to']] = connection['tie_points']

    # Generate the HTML table based on the matrix
    html_content = "<table border='1'>\n<tr><th>Image ID</th>"
    for col_id in sorted(image_ids):
        html_content += f"<th>{col_id}</th>"
    html_content += "</tr>\n"

    for row_id in sorted(image_ids):
        html_content += f"<tr><td>{row_id}</td>"
        for col_id in sorted(image_ids):
            tie_points =tie_points_matrix[row_id][col_id]
            html_content += f"<td>{tie_points}</td>"
        html_content += "</tr>\n"
    html_content += "</table>\n"

    return html_content

def _parse_homol_directory(homol_folder):
    """
    Parses the homol folder to find connections and tie-points between images.
    Returns a list of dictionaries with the connection data.
    """
    connections = []
    for folder_name in os.listdir(homol_folder):
        if folder_name.startswith("PastisOIS-Reech_"):
            image_id = folder_name.split("_")[-1].split(".")[0]
            for file_name in os.listdir(os.path.join(homol_folder, folder_name)):
                if file_name.startswith("OIS-Reech_") and file_name.endswith(".txt"):
                    connected_image_id = file_name.split("_")[-1].split(".")[0]
                    with open(os.path.join(homol_folder, folder_name, file_name), 'r') as file:
                        tie_points = len(file.readlines())  # Assuming one tie-point per line
                        connections.append({
                            'image_id': image_id,
                            'connected_to': connected_image_id,
                            'tie_points': tie_points
                        })
    return connections


if __name__ == "__main__":
    prj_fldr = "/data_1/ATM/data_1/sfm/projects/src_test2"

    update_html(prj_fldr)