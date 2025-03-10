# Required for streamlit
import sys
from pathlib import Path
src_path = (Path(__file__).parent.parent / 'src').resolve()
if str(src_path) not in sys.path:
    sys.path.append(str(src_path))

# Library imports
import math
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import streamlit as st

# Must be called before any Streamlit UI elements
st.set_page_config(layout="wide")

# Local imports
import base.connect_to_database as ctd  # noqa

tables_to_check = ["All tables",
                   "images", "images_extracted",
                   "images_fid_points", "images_file_paths",
                   "images_georef", "images_segmentation"]

debug_more_details = False
debug_max_cutoff = 100


def verify_psql(table, only_complete, conn, ignore_failed=False):
    # special handling for multiple tables
    if table == "All tables":
        table_results = compare_multiple_tables(conn)
    else:
        # create dict for specific table
        table_results = {
            "double_entries": {"count": 0, "image_ids": []},
            "missing_entries": {},
            "total_entries": 0,
        }

        # Check for double entries with filtering if needed
        if table == "images_georef" and ignore_failed:
            sql_string = (f"SELECT image_id FROM {table} "
                          "WHERE georef_type <> 'failed' "
                          "GROUP BY image_id HAVING COUNT(image_id) > 1")
        else:
            sql_string = f"SELECT image_id FROM {table} GROUP BY image_id HAVING COUNT(image_id) > 1"
        data = ctd.execute_sql(sql_string, conn)
        if not data.empty:
            table_results["double_entries"] = {
                "count": data.shape[0],
                "image_ids": data['image_id'].tolist()
            }

        # Get table data with filtering if needed
        if table == "images_georef" and ignore_failed:
            data = ctd.execute_sql(f"SELECT * FROM {table} WHERE georef_type <> 'failed'", conn)
        else:
            data = ctd.execute_sql(f"SELECT * FROM {table}", conn)

        # Filter for only complete flight-paths if requested
        if only_complete:
            data = filter_complete(data, conn)

        # Get number of total entries
        table_results["total_entries"] = len(data)

        # Identifying missing entries for each column
        for col in data.columns:
            if col in ["image_id", "comment", "last_change"]:
                continue

            missing = data[data[col].isnull()]['image_id'].tolist()
            percentage = (round(len(missing) / table_results["total_entries"] * 100, 2)
                          if table_results["total_entries"] > 0 else 0)

            table_results["missing_entries"][col] = {
                "count": len(missing),
                "percentage": percentage,
                "image_ids": missing,
            }

    return table_results

def compare_multiple_tables(conn):
    table_data = {}
    # iterate all tables except the first one
    for table in tables_to_check[1:]:
        sql_string = "SELECT image_id FROM " + table
        data = ctd.execute_sql(sql_string, conn)
        data = data['image_id'].tolist()
        # get all image_ids from the table
        table_data[table] = data

    # identify complete set of unique image_ids
    all_ids = set().union(*table_data.values())
    return_dict = {}
    return_dict["all_ids"] = len(all_ids)
    return_dict["missing_ids"] = {}

    for table, ids in table_data.items():
        missing_ids = all_ids - set(ids)
        return_dict["missing_ids"][table] = len(missing_ids)

    return return_dict

def filter_complete(data, conn):
    has_fl = False
    if 'flight_path' in data.columns:
        has_fl = True

    # get flight path based on image_ids if not present
    if not has_fl:
        data['flight_path'] = data['image_id'].str[2:6]

    flight_paths = data['flight_path'].unique()

    # get all flight path data from sql
    sql_string = "SELECT * FROM flight_paths"
    fl_data = ctd.execute_sql(sql_string, conn)
    fl_data['flight_path'] = fl_data['flight_path'].astype(int).astype(str).str.zfill(4)

    # Create a dictionary to hold the min/max image numbers from data
    flight_path_ranges = (
        data.groupby('flight_path')['image_id']
        .apply(lambda x: (x.str[-4:].astype(int).min(), x.str[-4:].astype(int).max()))
        .to_dict()
    )

    # Filter out incomplete flight paths
    complete_paths = []
    for flight_path in flight_paths:
        ref_min = fl_data.loc[fl_data['flight_path'] == flight_path, 'min_img_nr'].values[0]
        ref_max = fl_data.loc[fl_data['flight_path'] == flight_path, 'max_img_nr'].values[0]
        data_min, data_max = flight_path_ranges.get(flight_path, (None, None))
        if data_min == ref_min and data_max == ref_max:
            complete_paths.append(flight_path)

    # Keep only the rows with complete flight paths
    filtered_data = data[data['flight_path'].isin(complete_paths)]

    # remove flight_path column if it was added
    if not has_fl:
        filtered_data = filtered_data.drop(columns=['flight_path'])

    return filtered_data

def plot_results():
    # establish connection to psql
    conn = ctd.establish_connection()

    st.title("Database Tables Quality Control")

    # Allow selection of tables dynamically based on the results
    selected_table = st.sidebar.selectbox("Select table", tables_to_check)

    # checkbox for only complete flight-paths
    if selected_table != "All tables":
        only_complete = st.sidebar.checkbox("Only complete flight-paths")
    else:
        only_complete = False  # Default when checkbox is hidden

    # Add checkbox for 'ignore failed' if the selected table is images_georef
    if selected_table == "images_georef":
        ignore_failed = st.sidebar.checkbox("Ignore failed")
    else:
        ignore_failed = False

    # Pass ignore_failed to the verification function
    table_results = verify_psql(selected_table, only_complete, conn, ignore_failed)

    # -- If multiple tables: ---------------------------------------------------
    if selected_table == "All tables":
        st.header("Multiple Tables")
        st.write(f"**Sum of unique Image IDs across all tables:** {table_results['all_ids']}")
        n = len(table_results["missing_ids"])
        cols_in_grid = 5
        rows_in_grid = math.ceil(n / cols_in_grid)
        fig, axs = plt.subplots(rows_in_grid, cols_in_grid, figsize=(15, rows_in_grid * 3))
        fig.subplots_adjust(hspace=0.4, wspace=0.4)
        axs = axs.ravel()

        i = 0
        for table_name, missing_count in table_results["missing_ids"].items():
            fill_rate = 100 - (missing_count / table_results["all_ids"] * 100)
            color = mcolors.to_hex([1 - fill_rate / 100, fill_rate / 100, 0])
            axs[i].bar([0], [fill_rate], color=color)
            axs[i].set_ylim(0, 100)
            axs[i].set_xticks([0])
            axs[i].set_xticklabels([table_name])
            axs[i].set_title(f"{table_name}\nFilled: {fill_rate:.1f}%")
            i += 1

        for ax in axs[n:]:
            ax.set_visible(False)

        plt.tight_layout()
        st.pyplot(fig)
        return

    # -- If a single table is selected: ---------------------------------------
    st.header(f"Analysis of Table: {selected_table}")
    total_entries = table_results["total_entries"]
    st.write(f"**Total entries in '{selected_table}'**: {total_entries}")

    st.subheader("Double Entries")
    double_entries_count = table_results["double_entries"]["count"]
    if double_entries_count > 0:
        st.write(f"There are {double_entries_count} double entries.")
        if debug_more_details:
            image_ids = table_results["double_entries"]["image_ids"][:debug_max_cutoff]
            st.write("Example double entry IDs:", image_ids)
            if len(image_ids) == debug_max_cutoff:
                st.write(f"List was cut off after {debug_max_cutoff} entries.")
    else:
        st.write("There are no double entries.")

    st.subheader("Missing Entries")
    n = len(table_results["missing_entries"])
    cols_in_grid = 5
    rows_in_grid = math.ceil(n / cols_in_grid)
    fig, axs = plt.subplots(rows_in_grid, cols_in_grid, figsize=(15, rows_in_grid * 3))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    axs = axs.ravel()

    for i, col in enumerate(table_results["missing_entries"]):
        col_missing_info = table_results["missing_entries"][col]
        fill_rate = 100 - col_missing_info["percentage"]
        color = mcolors.to_hex([1 - fill_rate / 100, fill_rate / 100, 0])
        axs[i].bar([0], [fill_rate], color=color)
        axs[i].set_ylim(0, 100)
        axs[i].set_xticks([0])
        axs[i].set_xticklabels([col])
        axs[i].set_title(f"{col}\nFilled: {fill_rate:.1f}%")
        missing_count = col_missing_info["count"]
        axs[i].text(
            0, fill_rate / 2,
            f"{missing_count} missing\n({col_missing_info['percentage']}%)",
            ha='center', va='center', fontsize=8, color='black'
        )

    for ax in axs[n:]:
        ax.set_visible(False)

    plt.tight_layout()
    st.pyplot(fig)

if __name__ == "__main__":
    plot_results()
