import os
import pandas as pd

PATH_PROJECTS_FOLDER = "/data/ATM/data_1/sfm/agi_projects"


def get_all_finished():

    rows = []
    # iterate over all project folders
    for entry in os.listdir(PATH_PROJECTS_FOLDER):
        project_fld = os.path.join(PATH_PROJECTS_FOLDER, entry)

        # get project name
        project_name = project_fld.split("/")[-1]
        print("Collect data from", project_name)

        # define the output folder
        output_dir = os.path.join(project_fld, "output")

        # check if the project was finished with the absolute ortho
        if not os.path.exists(os.path.join(output_dir, project_name + "_ortho_absolute.tif")):
            continue

        # add the project to the output data
        rows.append({"project": project_name})

    finished_projects = pd.DataFrame(rows)
    return finished_projects

def get_data_from_table(project_name):

    sql_string = "SELECT * FROM sfm_projects WHERE project_name=''


if __name__ == "__main__":
    data = get_all_finished()

    print(data)