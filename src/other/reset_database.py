import base.connect_to_db as ctd

databases = ["images", "images_extracted", "images_fid_points"]


def reset_database():
    """
    reset_database():
    This function deletes all entries from the selected databases.
    WARNING: THIS CANNOT BE UNDONE
    Args:
    None
    Return:
    None
    """

    # ask if the user is really sure that he wants to do this
    user_input = input('Do you really want to reset the database (y/n): ')

    # only if the user states yes we are deleting
    if user_input != "y":
        print("Nothing was deleted")
    else:

        # the actual deleting
        for db in databases:
            sql_string = f"DELETE FROM {db}"
            ctd.edit_data_in_db(sql_string, add_timestamp=False, catch=False, verbose=True)


if __name__ == "__main__":
    reset_database()
