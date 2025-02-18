import src.load.load_image as li
import src.text.extract_altimeter as ea

from tqdm import tqdm

def apply_height_detection(image_id):

    try:
        img = li.load_image(image_id)

        altitude = ea.extract_altimeter(img)

        if altitude is not None:
            print(image_id, altitude)

            # attach to text file
            #with open("results3.txt", "a") as f:
            #    f.write(f"{image_id},{altitude}\n")
    except Exception as e:
        raise e
        return


if __name__ == "__main__":


    import src.base.connect_to_database as ctd
    sql_string = "SELECT image_id, altimeter_height FROM images_extracted WHERE substring(image_id, 3, 4) in ('2140', '2073', '1822', '1827', '1684', '2142', '1824', '1846', '2139', '2075', '1821', '1816', '1833', '2137', '1825', '2136', '2143', '1826', '1813', '2141')"
    import src.base.connect_to_database as ctd
    conn = ctd.establish_connection()
    data = ctd.execute_sql(sql_string, conn)

    # shuffle dataframe
    data = data.sample(frac=1).reset_index(drop=True)

    import numpy as np

    # load all ids from text file
    with open("results3.txt", "r") as f:
        lines = f.readlines()
        ids = [line.split(",")[0] for line in lines]

    for i, row in tqdm(data.iterrows(), total=data.shape[0]):

        #if np.isnan(row['altimeter_value']) is False:
        #    print("Skipping", row['image_id'], row['altimeter_value'])
        #    continue

        if row['image_id'] != 'CA182132V0042':
            continue

        if row['image_id'] not in ids:
            print("Skipping", row['image_id'])
            continue

        print("Checking", row['image_id'])
        apply_height_detection(row['image_id'])
