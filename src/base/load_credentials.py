"""load credentials from file"""

# Package imports
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
        tuple[str, str]: A tuple the username and password for the specified account ID.

    Raises:
        KeyError: If a key is not found in the credentials file.
    """

    # Open and read the JSON file
    with open(PATH_CREDENTIALS, 'r') as file:
        credentials_data = json.load(file)

        # Retrieve credentials by account ID
        if account_id in credentials_data:

            account = credentials_data[account_id]

            username = account['username']
            password = account['password']

            return username, password

        else:
            raise KeyError(f"There is no account '{account_id}' in the credentials file.")
