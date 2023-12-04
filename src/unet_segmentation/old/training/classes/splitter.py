import torch.utils

from skmultilearn.model_selection import IterativeStratification

import base.connect_to_db as ctd

class ImageSplitter():

    def __init__(self, fileIds, percentages, split_type, max_images, seed=123):

        if max_images is None:
            self.fileIds = fileIds
        else:
            self.fileIds = fileIds[:max_images]

        self.percentages = percentages
        self.split_type = split_type

        self.train_indexes = []
        self.train_ids = []
        self.val_indexes = []
        self.val_ids = []
        self.test_indexes = []
        self.test_ids = []

        self.labels = None

        # count the labels
        self.get_labels()

        if max_images is not None:
            self.labels = self.labels[:max_images,:]

        self.seed = seed

        if self.split_type == "random":
            self.split_random()
        elif split_type == "stratified":
            self.split_stratified()

        # check how the classes are distributed
        self.print_split()

    def get_labels(self):

        # get images that are hand-labelled!
        sql_string = "SELECT image_id, perc_ice as ice, perc_snow as snow, perc_rocks as rocks," \
                     "perc_water as water, perc_clouds as clouds, perc_sky as sky," \
                     "perc_other as other FROM images_segmentation " \
                     "WHERE labelled_by='manual'"

        data = ctd.get_data_from_db(sql_string)

        data.loc[data['ice'] > 0, 'ice'] = 1
        data.loc[data['snow'] > 0, 'snow'] = 1
        data.loc[data['rocks'] > 0, 'rocks'] = 1
        data.loc[data['water'] > 0, 'water'] = 1
        data.loc[data['clouds'] > 0, 'clouds'] = 1
        data.loc[data['sky'] > 0, 'sky'] = 1
        data.loc[data['other'] > 0, 'other'] = 1

        data = data.drop('image_id', axis=1)

        data = data.to_numpy()

        self.labels = data

    def split_stratified(self):

        n_splits = 2

        samples = [(self.percentages[1] + self.percentages[2])/100, self.percentages[0]/100]

        # no seed required as no shuffeling
        stratifier = IterativeStratification(n_splits=n_splits, sample_distribution_per_fold=samples)

        print(self.fileIds)
        print(self.labels)

        self.train_indexes, self.temp_indexes = next(stratifier.split(X=self.fileIds, y=self.labels))

        if self.percentages[2] == 0:
            self.val_indexes = self.temp_indexes
        else:

            temp_fileIds = []
            for i, elem in enumerate(self.fileIds):
                if i in self.temp_indexes:
                    temp_fileIds.append(elem)

            temp_labels = self.labels[self.temp_indexes]

            multiplier = 1/samples[0]
            samples_2 = [self.percentages[2]*multiplier/100, self.percentages[1] * multiplier/100]

            stratifier2 = IterativeStratification(n_splits=n_splits, sample_distribution_per_fold=samples_2)

            # necessary because sometimes it failed but works usually
            while True:
                try:
                    self.val_indexes, self.test_indexes = next(stratifier2.split(X=temp_fileIds, y=temp_labels))
                except:
                    continue
                break

        for elem in self.train_indexes:
            self.train_ids.append(self.fileIds[elem])

        for elem in self.val_indexes:
            self.val_ids.append(self.fileIds[elem])

        for elem in self.test_indexes:
            self.test_ids.append(self.fileIds[elem])

    def split_random(self):

        train_size = self.percentages[0]
        val_size = self.percentages[1]
        test_size = self.percentages[2]

        torch.manual_seed(self.seed)

        self.train_ids, self.val_ids, self.test_ids = torch.utils.data.random_split(
            self.fileIds, (train_size, val_size, test_size))

    def get_splitted(self):

        return self.train_ids, self.val_ids, self.test_ids

    def print_split(self):

        print("train-images: {}".format(self.train_indexes.shape[0]))
        print(self.labels[self.train_indexes].sum(axis=0))

        print("val-images: {}".format(self.val_indexes.shape[0]))
        print(self.labels[self.val_indexes].sum(axis=0))

        if self.percentages[2] > 0:
            print("test-images: {}".format(self.test_indexes.shape[0]))
            print(self.labels[self.test_indexes].sum(axis=0))

        print("----------")