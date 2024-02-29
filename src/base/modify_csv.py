import csv
import os
import tempfile
import shutil

from typing import Dict, Optional


def modify_csv(file_path: str, image_id: str, modus: str, data: Optional[Dict[str, str]] = None,
               overwrite: bool = False) -> Optional[bool]:
    """
    Modify a CSV file to add, delete, or check for an image_id and associated data.
    Creates the CSV file if it does not exist.

    Args:
        file_path (str): The path to the CSV file.
        image_id (str): The image_id to be added, deleted, or checked.
        modus (str): Operation mode - either 'add', 'delete', or 'check'.
        data (dict, optional): Additional data associated with the image_id. Defaults to None.
        overwrite (bool, optional): If True, allows overwriting existing data when adding. Defaults to False.

    Returns:
        Optional[bool]: Returns None for 'add' and 'delete' operations.
            For 'check' operation, returns True if image_id exists, otherwise False.
    """

    # define some parameters
    rows = []
    existing_columns = ["id"]

    # check if the file is already existing
    file_exists = os.path.exists(file_path)

    # Read existing data from the file if it exists
    if file_exists:
        with open(file_path, mode='r', newline='') as f:
            reader = csv.DictReader(f, delimiter=';')
            existing_columns.extend(reader.fieldnames)  # Extend the list with existing field names
            existing_columns = list(dict.fromkeys(existing_columns))  # Remove duplicates while preserving order
            for row in reader:
                rows.append(row)

    # Handle 'check' mode directly
    if modus == "check":
        return any(row["id"] == image_id for row in rows)

    # For 'add' mode, add or update data
    elif modus == 'add':
        if any(row["id"] == image_id for row in rows) and overwrite:
            # Update existing data if overwrite is True
            rows = [{**row, **data} if row["id"] == image_id else row for row in rows]
        else:
            # Add new row with data if image_id not found
            new_row = {"id": image_id, **data}
            rows.append(new_row)

            # Ensure new columns are added at the end
            for key in data.keys():
                if key not in existing_columns:
                    existing_columns.append(key)

    # For 'delete' mode, remove the specified image_id row
    elif modus == "delete":
        rows = [row for row in rows if row["id"] != image_id]

    # Normalize columns for all rows and ensure all rows have the same columns
    for row in rows:
        for col in existing_columns:
            row.setdefault(col, None)

    # Use a temporary file to safely write data
    temp_file = tempfile.NamedTemporaryFile(delete=False)

    try:
        # write data to the temporary file
        with open(temp_file.name, 'w', newline='') as tf:
            writer = csv.DictWriter(tf, fieldnames=existing_columns, delimiter=';')
            writer.writeheader()
            writer.writerows(rows)

        # Replace the original file with the updated one
        shutil.move(temp_file.name, file_path)

    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file.name):
            os.remove(temp_file.name)

    return None
