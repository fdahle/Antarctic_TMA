import matplotlib.pyplot as plt
import src.base.connect_to_database as ctd
import numpy as np
import pandas as pd

flight_paths = ["1821", "1816", "1833", "2137", "1825", "2136",
                       "2143", "1826", "1813", "2141",
                "2140", "2073", "1822", "1827", "1684", "2142",
                       "1824", "1846", "2139", "2075"]


def estimate_altitude(flight_path, conn):
    sql_string = ("SELECT image_id, altimeter_value "
                  "FROM images_extracted WHERE "
                  "altimeter_estimated is False and "
                  f"substring(image_id, 3, 4)='{flight_path}'")
    data = ctd.execute_sql(sql_string, conn)

    # Remove NaN or None values
    data = data.dropna(subset=['altimeter_value'])

    # Compute weighted mean after removing outliers
    weighted_mean, filtered_df = calculate_weighted_mean(data)

    if weighted_mean is None:
        return

    # round weighted mean to 100 steps
    weighted_mean = round(weighted_mean, -3)

    # update the database
    sql_string = (f"UPDATE images_extracted SET "
                  f"altimeter_value={weighted_mean},"
                  f"altimeter_estimated=True "
                  f"WHERE SUBSTRING(image_id, 3, 4)='{flight_path}' "
                  f"AND altimeter_estimated IS NULL")
    print(sql_string)
    ctd.execute_sql(sql_string, conn)

    return

    print(flight_path)
    print("Unique altimeter values and their percentage:")
    print(filtered_df)
    print(f"Weighted Mean Altimeter Value: {weighted_mean}")

    # Extract image number (last 4 characters of image_id)
    data['image_nr'] = data['image_id'].str[-4:].astype(int)

    # Sort data by image number
    data = data.sort_values(by='image_nr')

    # Fit a linear regression line
    if len(data) > 1:
        coefficients = np.polyfit(data['image_nr'], data['altimeter_value'], 1)
        poly = np.poly1d(coefficients)
        fitted_y = poly(data['image_nr'])

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(data['image_nr'], data['altimeter_value'], marker='o', linestyle='-', label=f'Flight {flight_path}')
    if len(data) > 1:
        plt.plot(data['image_nr'], fitted_y, linestyle='--', color='red', label='Linear Fit')
    plt.axhline(weighted_mean, color='green', linestyle='-.', label='Weighted Mean')
    plt.xlabel('Image Number')
    plt.ylabel('Altimeter Value')
    plt.title(f'Altimeter Value vs Image Number for Flight {flight_path}')
    plt.legend()
    plt.grid(True)
    plt.show()


def calculate_weighted_mean(data):
    """
    Calculate the weighted mean of altimeter value after removing outliers.
    """
    # Compute value counts
    value_counts = data['altimeter_value'].value_counts(normalize=True) * 100
    unique_vals_df = pd.DataFrame({'altimeter_value1': value_counts.index, 'Percentage': value_counts.values})

    # Detect and remove outliers using IQR
    Q1 = unique_vals_df['altimeter_value1'].quantile(0.25)
    Q3 = unique_vals_df['altimeter_value1'].quantile(0.75)
    IQR = Q3 - Q1

    # Define outlier thresholds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filter out outliers
    filtered_df = unique_vals_df[
        (unique_vals_df['altimeter_value1'] >= lower_bound) &
        (unique_vals_df['altimeter_value1'] <= upper_bound)
        ]

    if filtered_df.shape[0] == 0:
        return None, None

    # Compute weighted mean
    weighted_mean = np.average(filtered_df['altimeter_value1'], weights=filtered_df['Percentage'])

    return weighted_mean, filtered_df


if __name__ == "__main__":
    conn = ctd.establish_connection()
    for flight_path in flight_paths:
        estimate_altitude(flight_path, conn)
