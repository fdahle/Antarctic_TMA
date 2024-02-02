import psycopg2
from psycopg2 import OperationalError

def test_postgresql_connection(host, port, database, user, password):
    try:
        print("Start testing")
        # Try to establish a connection to the database
        connection = psycopg2.connect(
            host=host,
            port=port,
            database=database,
            user=user,
            password=password
        )
        
        # Create a cursor object
        cursor = connection.cursor()
        
        # Execute a simple SQL query to test the connection
        cursor.execute("SELECT 1;")
        
        # Fetch the result to make sure the query was executed successfully
        cursor.fetchone()
        
        # Close the cursor and connection
        cursor.close()
        connection.close()
        
        print("Connection to PostgreSQL database was successful.")
    except OperationalError as e:
        print(f"An error occurred: {e}")
        print("Failed to connect to PostgreSQL database.")

# Example usage
test_postgresql_connection(
    host='localhost',
    port='7777',  # Local forwarded port
    database='antarctica2',
    user='admin',
    password='password'
)
