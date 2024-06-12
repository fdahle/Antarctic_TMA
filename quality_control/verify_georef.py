# Required for streamlit
import copy
import sys
from pathlib import Path
src_path = (Path(__file__).parent.parent / 'src').resolve()
if str(src_path) not in sys.path:
    sys.path.append(str(src_path))

# Library imports
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# Local imports
import base.connect_to_database as ctd  # noqa

BASE_FLD = "/data_1/ATM/data_1/georef"


def compute_pie(data):
    # remove all entries where 'status' is empty
    data_filled = data.dropna(subset=['status'])

    # count the different status
    status_count = data_filled['status_reason'].value_counts().to_dict()

    return status_count

def compute_statistics(status_type, data):

    # remove all entries where 'status' is empty
    data_filled = data.dropna(subset=['status'])

    if status_type != "All":
        data = data_filled[data_filled['status_reason'] == status_type]
    else:
        data = data_filled


    metrics = ["num_tps", "confidence", "residuals", "complexity"]
    stats = {}
    for metric in metrics:

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
        "confidence": "Confidence",
        "residuals": "Residuals",
        "complexity": "Complexity"
    }
    stats_df.rename(columns=new_column_names, inplace=True)

    # Change data type of TPs to integer
    try:
        stats_df['Tps'] = stats_df['Tps'].astype(int)
    except (Exception,):
        # just do nothing
        pass

    return stats_df

def compute_flight_statistics(data, flight_path):

    # filter by flight path
    data_per_flight = data[data['tma_number'] == int(flight_path)]

    # get matrix of method and status_reason
    result_pivot = data_per_flight.pivot_table(index='method', columns='status_reason', aggfunc='size', fill_value=0)

    return result_pivot


def prepare_data():

    # load the shapefiles with additional information
    shp_data_sat = gpd.read_file(BASE_FLD + "/" + "sat.shp")
    #  shp_data_img = gpd.read_file(BASE_FLD + "/" + img.shp)
    # shp_data_calc = gpd.read_file(BASE_FLD + "/" + "calc.shp")
    # shp_data = pd.concat([shp_data_sat, shp_data_img, shp_data_calc], ignore_index=True)
    shp_data = shp_data_sat

    # load images sql data
    conn = ctd.establish_connection()
    sql_string = "SELECT image_id, tma_number FROM images"
    data_images = ctd.execute_sql(sql_string, conn)

    # load images_extracted sql data
    sql_string = "SELECT image_id, complexity FROM images_extracted"
    data_images_extracted = ctd.execute_sql(sql_string, conn)

    # merge the sql_data
    sql_data = data_images.merge(data_images_extracted, on="image_id", how="left")

    # load all csv data
    csv_data_sat = pd.read_csv(BASE_FLD + "/" + "sat_processed_images.csv", delimiter=";")
    csv_data_img = pd.read_csv(BASE_FLD + "/" + "img_processed_images.csv", delimiter=";")
    csv_data_calc = pd.read_csv(BASE_FLD + "/" + "calc_processed_images.csv", delimiter=";")
    csv_data = pd.concat([csv_data_sat, csv_data_img, csv_data_calc], ignore_index=True)

    # merge all shp data into one dataframe
    data = sql_data.merge(shp_data, on="image_id", how="left")

    # merge all csv data into one dataframe
    data = data.merge(csv_data, left_on="image_id", right_on='id', how="left")

    # remove the id and geom column
    data.drop(columns=['id', 'geometry'], inplace=True)

    # rename some columns
    data.rename(columns={"avg_conf": "confidence"}, inplace=True)
    data.rename(columns={"avg_resi": "residuals"}, inplace=True)

    # remove all images that have no method
    data = data.dropna(subset=['method'])

    # add nr of tps for the entries "too_few_tps"
    condition = data['reason'].str.startswith('too_few_tps', na=False)
    data.loc[condition, 'num_tps'] = data['reason'].str.extract(r'too_few_tps:(\d+)', expand=False).astype(float)

    # Simplify 'reason' by removing details'
    data['reason'] = data['reason'].str.replace(r'too_few_tps:\d+', 'too_few_tps', regex=True)
    data['reason'] = data['reason'].str.replace(r'failed:\d+', 'failed', regex=True)

    # Create a new column that combines 'status' and 'reason' for the case where status is 'failed'
    data['status_reason'] = data.apply(
        lambda x: f"{x['status']} - {x['reason']}" if x['status'] == 'failed' else x['status'], axis=1)

    # cast tma number to integer
    data['tma_number'] = data['tma_number'].astype(int)

    return data

def plot_results(data):

    georef_to_check = ["all", "sat", "img", "calc"]

    st.title("Georef Quality Control")

    # allow selection of different georef types
    georef_type = st.selectbox("Select georef type", georef_to_check)

    # determine which pandas dataframe to use
    if georef_type == "all":
        # nothing needs to be done
        pass
    elif georef_type == "sat":
        # filter for status 'sat'
        data = data[data['status'] == 'sat']
    elif georef_type == "img":
        data = data[data['status'] == 'img']
    elif georef_type == "calc":
        data = data[data['status'] == 'calc']
    else:
        raise ValueError("Invalid georef type")

    # pie-chart for status
    st.header("Status")

    # get data for pie-chart
    pie_plot_data = compute_pie(data)

    # plot the pie-chart
    fig, ax = plt.subplots()
    ax.pie(pie_plot_data.values(),
           labels=pie_plot_data.keys(),
           autopct='%1.1f%%', startangle=90)
    ax.axis('equal')  # Ensures that pie is drawn as a circle.
    st.pyplot(fig)

    # statistical values for the data
    st.header("Average values")

    # possible to select the status type for which the statistics should be computed
    status_labels = ["All"] + list(pie_plot_data.keys())
    status_type = st.selectbox("Select status type", status_labels)

    # calculate statistics for the selected status type
    status_statistics = compute_statistics(status_type, data)

    # display statistics in the table
    st.table(pd.DataFrame(status_statistics))

    st.header("Flight paths")

    # get unique flight paths
    flight_paths = list(data['tma_number'].unique())

    # sort and convert to strings
    flight_paths = sorted(flight_paths)
    flight_paths = [str(int(f)) for f in flight_paths]

    flight_path = st.selectbox("Select flight path", flight_paths)

    # compute statistics for the selected flight path
    flight_statistics = compute_flight_statistics(data, flight_path)

    # display statistics in the table
    st.table(pd.DataFrame(flight_statistics))

    # show the raw data
    st.header("Raw data")
    st.dataframe(data)

if __name__ == "__main__":

    data = prepare_data()

    plot_results(data)
