# Required for streamlit
import copy
import sys
from pathlib import Path
src_path = (Path(__file__).parent.parent / 'src').resolve()
if str(src_path) not in sys.path:
    sys.path.append(str(src_path))

# Package imports
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# Custom imports
import base.connect_to_database as ctd  # noqa

BASE_FLD = "/data_1/ATM/data_1/georef"


def compute_statistics(data):

    metrics = ["num_tps", "avg_conf", "avg_resi", "complexity"]
    stats = {}
    for metric in metrics:
        print(metric)
        stats[metric] = {
            'mean': data[metric].mean(),
            'median': data[metric].median(),
            'min': data[metric].min(),
            'max': data[metric].max()
        }

    # Convert stats dictionary to a DataFrame
    stats_df = pd.DataFrame(stats)  # Transpose to make metrics the columns

    # Rename columns for clarity in presentation
    new_column_names = {
        "num_tps": "Tps",
        "avg_conf": "Confidence",
        "avg_resi": "Residuals",
        "complexity": "Complexity"
    }
    stats_df.rename(columns=new_column_names, inplace=True)

    # Change data type of TPs to integer
    stats_df['Tps'] = stats_df['Tps'].astype(int)

    return stats_df

def compute_flight_statistics(data, flight_path):

    data_fl = copy.deepcopy(data)

    # filter by flight path



def verify_georef():

    # load the shapefiles with additional information
    sat_shp_data = gpd.read_file(BASE_FLD + "/" + "sat.shp")

    # load psql data
    conn = ctd.establish_connection()
    sql_string = "SELECT image_id, tma_number FROM images"
    data_images = ctd.execute_sql(sql_string, conn)

    sql_string = "SELECT image_id, complexity FROM images_extracted"
    data_images_extracted = ctd.execute_sql(sql_string, conn)

    sql_data = data_images.merge(data_images_extracted, on="image_id", how="left")

    # load all csv files with information
    sat_data = pd.read_csv(BASE_FLD + "/" + "sat_processed_images.csv", delimiter=";")
    img_data = pd.read_csv(BASE_FLD + "/" + "img_processed_images.csv", delimiter=";")
    calc_data = pd.read_csv(BASE_FLD + "/" + "calc_processed_images.csv", delimiter=";")

    georef_to_check = ["all", "sat", "img", "calc"]

    st.title("Georef Quality Control")

    # allow selection of different georef types
    georef_type = st.selectbox("Select georef type", georef_to_check)

    # determine which pandas dataframe to use
    if georef_type == "all":
        # append pandas dataframes
        data = pd.concat([sat_data, img_data, calc_data], ignore_index=True)
        # TODO ONLY KEEP GEOREF WHEN IDS ARE MULTIPLE TIMES IN THE DATAFRAME
    elif georef_type == "sat":
        data = sat_data
    elif georef_type == "img":
        data = img_data
    elif georef_type == "calc":
        data = calc_data
    else:
        raise ValueError("Invalid georef type")

    # replace column name
    data.rename(columns={"id": "image_id"}, inplace=True)

    # merge data with shapedata
    data = data.merge(sat_shp_data, on="image_id", how="left")

    # merge data with sql data
    data = data.merge(sql_data, on="image_id", how="right")

    # add nr of tps for the entries "too_few_tps"
    condition = data['reason'].str.startswith('too_few_tps', na=False)
    data.loc[condition, 'num_tps'] = data['reason'].str.extract(r'too_few_tps:(\d+)', expand=False).astype(float)

    # Simplify 'reason' by removing details'
    data['reason'] = data['reason'].str.replace(r'too_few_tps:\d+', 'too_few_tps', regex=True)
    data['reason'] = data['reason'].str.replace(r'failed:\d+', 'failed', regex=True)

    # Create a new column that combines 'status' and 'reason' for the case where status is 'failed'
    data['status_reason'] = data.apply(
        lambda x: f"{x['status']} - {x['reason']}" if x['status'] == 'failed' else x['status'], axis=1)

    # remove all entries where 'status' is empty
    data_filled = data.dropna(subset=['status'])

    # count the different status
    status_count = data_filled['status_reason'].value_counts().to_dict()

    # pie-chart for status
    st.header("Status")
    fig, ax = plt.subplots()
    ax.pie(status_count.values(),
           labels=status_count.keys(),
           autopct='%1.1f%%', startangle=90)
    ax.axis('equal')  # Ensures that pie is drawn as a circle.

    st.pyplot(fig)

    # statistical values for the data
    st.header("Average values")

    status_labels = ["All"] + list(status_count.keys())

    status_type = st.selectbox("Select status type", status_labels)
    if status_type != "All":
        data_for_statistics = data_filled[data_filled['status_reason'] == status_type]
    else:
        data_for_statistics = data_filled

    statistics = compute_statistics(data_for_statistics)

    st.table(pd.DataFrame(statistics))

    st.header("Flight paths")

    # get unique flight paths
    flight_paths = list(data['tma_number'].unique())

    # sort and convert to strings
    flight_paths = sorted(flight_paths)
    flight_paths = [str(int(f)) for f in flight_paths]

    flight_path = st.selectbox("Select flight path", flight_paths)

    st.header("Raw data")

    # remove geometry column
    raw_data = data.drop(columns=['geometry'])
    st.dataframe(raw_data)

if __name__ == "__main__":

    verify_georef()
