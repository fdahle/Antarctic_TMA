import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import src.base.connect_to_database as ctd

_flight_path = 1829


def visualize_flight_path(flight_path):

    conn = ctd.establish_connection()

    sql_string = f"SELECT image_id, altitude FROM images where tma_number= '{str(flight_path)}' and " \
                 f"view_direction = 'V' ORDER BY image_id"

    data = ctd.execute_sql(sql_string, conn)

    # Extract ordered ids and altitudes
    ordered_ids = [int(row[1]['image_id'][-4:]) for row in data.iterrows()]  # Convert to integers for plotting
    altitudes = [row[1]['altitude'] for row in data.iterrows()]

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(ordered_ids, altitudes, marker='o', linestyle='-', color='b')
    plt.title('Flight Path Altitude Profile')
    plt.xlabel('Order (Derived from Image ID)')
    plt.ylabel('Altitude')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    visualize_flight_path(_flight_path)