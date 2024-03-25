import sys
sys.path.append('/home/fdahle/Documents/GitHub/Antarctic_TMA')  # Adjust this path as necessary

import math
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import streamlit as st

import src.base.connect_to_database as ctd

tables_to_check = ["images", "images_extracted", "images_fid_points"]

debug_more_details = False
debug_max_cutoff = 100


def verify_psql(table):

    # establish connection to psql
    conn = ctd.establish_connection()

    # create dict for specific table
    table_results = {
        "double_entries": {"count": 0, "image_ids": []},
        "missing_entries": {},
        "total_entries": 0,
    }

    # check first for double entries
    sql_string = f"SELECT image_id FROM {table} GROUP BY image_id HAVING COUNT(image_id) > 1"
    data = ctd.execute_sql(sql_string, conn)
    if not data.empty:
        table_results["double_entries"] = {
            "count": data.shape[0],
            "image_ids": data['image_id'].tolist()
        }

    # Get table data
    data = ctd.execute_sql(f"SELECT * FROM {table}", conn)
    table_results["total_entries"] = len(data)

    # Identifying missing entries
    for col in data.columns:
        if col in ["image_id", "comment", "last_change"]:
            continue

        missing = data[data[col].isnull()]['image_id'].tolist()
        percentage = round(len(missing) / table_results["total_entries"] * 100, 2) if table_results[
                                                                                          "total_entries"] > 0 else 0

        table_results["missing_entries"][col] = {
            "count": len(missing),
            "percentage": percentage,
            "image_ids": missing,
        }

    return table_results


def plot_results():

    st.title("Database Tables Quality Control")

    # Allow selection of tables dynamically based on the results
    selected_table = st.sidebar.selectbox("Select table", tables_to_check)

    table_results = verify_psql(selected_table)

    st.header("Double Entries")

    if table_results["double_entries"]["count"] > 0:
        st.write(f"There are {table_results['double_entries']['count']} double entries in the '{selected_table}' table.")
        if debug_more_details:
            image_ids = table_results["double_entries"]["image_ids"][:debug_max_cutoff]
            st.write("Example double entry IDs:", image_ids)
            if len(image_ids) == debug_max_cutoff:
                st.write(f"List was cut off after {debug_max_cutoff} entries.")
    else:
        st.write(f"There are no double entries in the '{selected_table}' table.")

    st.header("Missing Entries")

    # Determine grid size
    n = len(table_results["missing_entries"])
    cols_in_grid = 5  # Number of columns in the grid
    rows_in_grid = math.ceil(n / cols_in_grid)

    fig, axs = plt.subplots(rows_in_grid, cols_in_grid, figsize=(15, rows_in_grid * 3))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)  # Adjust space between plots

    # Flatten the array of axes for easy iteration
    axs = axs.ravel()

    for i, col in enumerate(table_results["missing_entries"]):
        fill_rate = 100 - table_results["missing_entries"][col]["percentage"]

        # Calculate color based on fill_rate
        color = mcolors.to_hex([1 - fill_rate / 100, fill_rate / 100, 0])

        axs[i].bar(col, fill_rate, color=color)
        axs[i].set_ylim(0, 100)
        axs[i].set_title(col)
        axs[i].set_ylabel('% Filled')

        axs[i].annotate(f'{fill_rate}%',
                        xy=(0.5, fill_rate),  # Adjust x position to center
                        xytext=(0, 10),  # 10 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

        # Interactive element to reveal missing IDs
        #if st.button(f'Show missing IDs for {col}', key=f'button_{i}'):
        #    missing_ids = ', '.join(table_results["missing_entries"][col]["image_ids"])
        #    st.text_area(f"Missing IDs for {col}", missing_ids, height=100)

    # Hide any unused axes if the number of plots is not a perfect fill of the grid
    for ax in axs[n:]:
        ax.set_visible(False)

    st.pyplot(fig)

if __name__ == "__main__":
    plot_results()