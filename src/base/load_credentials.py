"""load credentials from file"""

# Library imports
import json

# Constants
PATH_CREDENTIALS = '../../credentials.json'


def load_credentials(account_id: str) -> tuple[str, str]:
    """
    Load credentials for a specific account ID from a JSON file.

    This function reads a JSON file named 'credentials.json' which contains credentials
    associated with account IDs. It then returns the credentials for the specified account ID.

    Args:
        account_id (str): The ID of the account for which to retrieve credentials.

    Returns:
        account_info (object): An object containing the retrieved credentials.

    Raises:
        KeyError: If a key is not found in the credentials file.
    """

    # Open and read the JSON file
    with open(PATH_CREDENTIALS, 'r') as file:
        credentials_data = json.load(file)

        # Retrieve credentials by account ID
        if account_id in credentials_data:

            # Get the account information
            account_info = credentials_data[account_id]

            return account_info

        else:
            raise KeyError(f"There is no account '{account_id}' in the credentials file.")
