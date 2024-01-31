import csv
import os


def modify_csv(file_path, image_id, modus, data=None, overwrite=False):
    """
    Modify CSV file to add or delete an image_id and associated data.

    Parameters:
    - file_path (str): path to the CSV file
    - image_id (str): the image_id to be added or deleted
    - modus (str): either 'add' or 'delete' for the operation
    - data (dict): additional data associated with the image_id
    """

    # Ensure the data parameter is a dictionary
    if data is None:
        data = {}

    rows = []
    existing_columns = {"id"}

    # Check if the file exists
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            reader = csv.DictReader(f, delimiter=';')
            for row in reader:
                rows.append(row)
                existing_columns.update(row.keys())

    if modus == "check":
        return any(row["id"] == image_id for row in rows)

    elif modus == 'add':
        # If image_id is already present and overwrite is True, update the data
        if any(row["id"] == image_id for row in rows) and overwrite:
            for row in rows:
                if row["id"] == image_id:
                    for key, value in data.items():
                        row[key] = value
        # If image_id is not already present, add it with associated data
        elif not any(row["id"] == image_id for row in rows):
            new_row = {"id": image_id}
            for key, value in data.items():
                new_row[key] = value
            rows.append(new_row)

            # Update existing columns with new data keys
            existing_columns.update(data.keys())

            # Ensure all rows have the same columns
            for row in rows:
                for col in existing_columns:
                    row.setdefault(col, None)

    elif modus == "delete":
        # Delete the row with the specified image_id
        rows = [row for row in rows if row["id"] != image_id]

        # Remove columns that have null values for all rows
        columns_to_remove = set()
        for col in existing_columns:
            if col != "id" and all(len(row.get(col, "")) == 0 for row in rows):
                columns_to_remove.add(col)
        existing_columns -= columns_to_remove

    # Ensure "id" is the first column
    columns_ordered = ["id"] + sorted([col for col in existing_columns if col != "id"])

    # Write back to the file
    with open(file_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns_ordered, delimiter=';')
        writer.writeheader()
        for row in rows:
            # Filter the row to only include columns that are in columns_ordered
            filtered_row = {k: v for k, v in row.items() if k in columns_ordered}
            writer.writerow(filtered_row)
    return None


if __name__ == "__main__":

    base_path = "/data_1/ATM/data_1/playground/georef3"
    path_failed_csv_no_tie_points = base_path + "/overview/sat_tie_points.csv"

    print(path_failed_csv_no_tie_points)

    # Test the function
    # a = modify_csv(path_failed_csv_no_tie_points, "CA23414", "add", data={'tie_points': 23})  # For adding an image_id
    a = modify_csv(path_failed_csv_no_tie_points, "13434", "delete", data={'tie_points': 43})
    print(a)
