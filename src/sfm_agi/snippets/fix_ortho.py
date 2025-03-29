"""fix ortho by setting the no data correct"""
import rasterio


def fix_ortho(path, new_nodata):

    # Open the source file in read mode
    with rasterio.open(path) as src:
        profile = src.profile.copy()
        profile.update(nodata=new_nodata)

        # Read the data into memory
        data = src.read()

    # Save with the updated NoData value
    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(data)


if __name__ == "__main__":

    glaciers = ["Aagaard Glacier", "Archer Glacier", "Attlee Glacier", "Bevin Glacier",
                "Bolton Glacier", "Centurian Glacier", "Eden Glacier", "Evans Glacier",
                "Forbes Glacier", "Getman Ice Piedmont", "Hurley Glacier", "Iliad Glacier",
                "Jones IS", "Leonardo Glacier", "Miethe Glacier", "Mitterling Glacier",
                "Morrison Glacier", "Nemo Glacier", "Neny Glacier", "Niepce Glacier",
                "Northeast Glacier", "Orel Ice Fringe", "Perutz Glacier", "Petzval Glacier",
                "Rachel Glacier", "Reid Glacier", "Sikorsky Glacier", "Snowshoe Glacier",
                "Somigliana Glacier", "Stubb Glacier", "Swithinbank Glacier", "Thunder Glacier",
                "William Glacier"
                ]

    for glac in glaciers:
        print("Fixing ortho for", glac)
        glac_name = glac.lower().replace(" ", "_")
        path = f"/data/ATM/data_1/sfm/agi_projects/{glac_name}/output/{glac_name}_ortho_absolute.tif"
        fix_ortho(path, 255)