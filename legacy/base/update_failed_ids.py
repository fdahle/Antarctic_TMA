import json
import os
import pandas as pd

from datetime import datetime


class FailManager:

    def __init__(self, workflow, save_every_step=False, csv_folder=None):

        # save the arguments
        self.workflow = workflow
        self.save_every_step = save_every_step

        # load the json to get default values
        json_folder = os.path.dirname(os.path.realpath(__file__))
        with open(json_folder + "/params.json") as j_file:
            json_data = json.load(j_file)

        if csv_folder is None:
            csv_folder = json_data["path_folder_failed_ids"]

        # define path to csv-file
        self.csv_path = csv_folder + "/" + workflow + ".csv"

        # read existing content
        if os.path.isfile(self.csv_path):
            self.data = pd.read_csv(self.csv_path, sep=";", header=0)

        # or create new pandas dataframe
        else:
            self.data = pd.DataFrame(columns=['image_id', 'description', 'datetime'])

    def update_failed_id(self, image_id, description):

        datetime_str = datetime.now().strftime("%d/%m/%Y, %H:%M")

        # image image_id is already in the dataframe
        if image_id in self.data["image_id"].values:

            # update the values
            self.data.loc[self.data['image_id'] == image_id, 'description'] = description
            self.data.loc[self.data['image_id'] == image_id, 'datetime'] = datetime_str

        # image image_id is not already in the dataframe
        else:

            # add to the dataframe
            row = pd.DataFrame({"image_id": image_id,
                                "description": description,
                                "datetime": datetime_str}, index=[0])

            self.data = pd.concat([self.data, row], ignore_index=True)

        if self.save_every_step:
            self.save_csv()

    def remove_failed_id(self, image_id):

        self.data = self.data.drop(self.data[self.data['image_id'] == image_id].index)

        if self.save_every_step:
            self.save_csv()

    def save_csv(self):

        # drop data to csv
        self.data.to_csv(self.csv_path, index=False, sep=";", header=True)
