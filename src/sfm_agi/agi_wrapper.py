# Library imports
import json
import os
import shutil
import sys
import subprocess
import tempfile

PATH_FOLDER_PROJECTS = "/data/ATM/data_1/sfm/agi_projects"


def agi_wrapper(project_name, images,
                camera_positions=None, camera_accuracies=None,
                focal_lengths=None):
    # Serialize data structures to pass as arguments
    images = json.dumps(images)
    camera_positions = json.dumps(camera_positions if camera_positions is not None else {})
    camera_accuracies = json.dumps(camera_accuracies if camera_accuracies is not None else {})
    focal_lengths = json.dumps(focal_lengths if focal_lengths is not None else {})

    # Path to the script containing the subprocess function
    script_path = '/home/fdahle/Documents/GitHub/Antarctic_TMA/src/sfm_agi/agi_wrapper.py.py'

    # Command to execute the Python script
    command = [
        sys.executable, script_path,
        project_name, images,
        camera_positions, camera_accuracies, focal_lengths
    ]

    # Create a temporary file to store logs
    temp_log_file = tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.log')

    # open a log file
    try:
        # Execute the command and capture output
        with subprocess.Popen(command,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE,
                              universal_newlines=True) as p:
            for stdout_line in p.stdout:
                print(stdout_line, end="")
                temp_log_file.write(stdout_line)

            for stdout_line in p.stderr:
                print(stdout_line, end="")
                temp_log_file.write(stdout_line)

    finally:
        temp_log_file.close()  # Ensure the file is closed before moving it
        # Define the final path for the log file (ensure the directory exists)
        final_log_dir = os.path.join(PATH_FOLDER_PROJECTS, project_name)
        final_log_path = os.path.join(final_log_dir, 'output.log')
        # Move the log file from the temporary location to the final destination
        shutil.move(temp_log_file.name, final_log_path)
