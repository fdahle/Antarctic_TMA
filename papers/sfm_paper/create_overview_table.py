import os
import pandas as pd

import src.base.connect_to_database as ctd

PATH_PROJECTS_FOLDER = "/data/ATM/data_1/sfm/agi_projects"


def get_all_finished():

    rows = []
    # iterate over all project folders
    for entry in os.listdir(PATH_PROJECTS_FOLDER):
        project_fld = os.path.join(PATH_PROJECTS_FOLDER, entry)

        # get project name
        project_name = project_fld.split("/")[-1]

        # define the output folder
        output_dir = os.path.join(project_fld, "output")

        # check if the project was finished with the absolute ortho
        if not os.path.exists(os.path.join(output_dir, project_name + "_ortho_absolute.tif")):
            continue

        # add the project to the output data
        rows.append({"project": project_name})

    finished_projects = pd.DataFrame(rows)
    return finished_projects

def get_data_from_table(project_name, conn=None):

    if conn is None:
        conn = ctd.establish_connection()

    sql_string = (f"SELECT num_images, st_area(area) / 1000000 as area, "
                  f"num_gcps, gcps_rmse, "
                  f"c_all_diff_abs_median, c_mask_diff_abs_median, "
                  f"c_all_rmse, c_mask_rmse, "
                  f"c_mask_mad, c_all_mad "
                  f"FROM sfm_projects WHERE project_name='{project_name}' "
                  f"AND status='finished' "
                  f"ORDER BY date_time DESC LIMIT 1"
                 )
    sql_data = ctd.execute_sql(sql_string, conn)

    if sql_data.shape[0] == 0:
        print("No data found for project", project_name)
        return None

    return sql_data

if __name__ == "__main__":
    dataframe = get_all_finished()

    all_data = []

    for idx, row in dataframe.iterrows():
        project_name = row['project']
        sql_data = get_data_from_table(project_name)

        if sql_data is not None:
            # merge project name into the row if it's not included already
            sql_data['project'] = project_name
            all_data.append(sql_data)

    # Combine all individual project data into a single DataFrame
    if all_data:
        full_df = pd.concat(all_data, ignore_index=True)
    else:
        print("No project data found.")
        exit()

    column_order = [
        "project",
        "area",
        "num_images",
        "c_all_diff_abs_median",
        "c_mask_diff_abs_median",
        "c_all_rmse",
        "c_mask_rmse",
        "c_all_mad",
        "c_mask_mad"
    ]

    # Reorder DataFrame columns
    full_df = full_df[column_order]

    # Cast num_images to int
    full_df["num_images"] = full_df["num_images"].astype(int)

    # Format float columns to 2 decimal places
    float_cols = [
        "area",
        "c_all_diff_abs_median",
        "c_mask_diff_abs_median",
        "c_all_rmse",
        "c_mask_rmse",
        "c_all_mad",
        "c_mask_mad"
    ]
    for col in float_cols:
        full_df[col] = full_df[col].map(lambda x: f"{x:.1f}")

    # Replace underscores with spaces and capitalize each word
    full_df["project"] = full_df["project"].str.replace("_", " ").str.title()

    # order by project name
    full_df = full_df.sort_values(by="project")

    latex_code = full_df.to_latex(index=False)
    print(latex_code)
