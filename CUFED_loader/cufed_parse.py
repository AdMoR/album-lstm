import json
import os
import itertools
from collections import defaultdict


class Cufed(object):

    def __init__(self, image_folder, label_path):
        self.image_folder = image_folder
        self.label_dict = json.load(open(label_path))
        self.prepare_classes_to_index()

    def class_stats(self):
        all_classes = defaultdict(lambda :0)
        for v in self.label_dict.values():
            if type(v) == list:
                for vv in v:
                    all_classes[vv] += 1
            else:
                all_classes[v] += 1

        return all_classes

    def prepare_classes_to_index(self):
        self.label_to_index = dict()
        label_index = 0
        per_label_ids = defaultdict(list)

        # Assign a label index for each label
        for k, v in self.label_dict.items():
            if type(v) == list:
                v = v[0]
            if v not in self.label_to_index.keys():
                self.label_to_index[v] = label_index
                label_index += 1
            # Add the album id to the list of this label
            per_label_ids[self.label_to_index[v]].append(k)

        # Now define the training and validation for the set of albums
        self.training_albums = list(itertools.chain(*[l[:-3] for l in per_label_ids.values()]))
        self.validation_albums = list(itertools.chain(*[l[-3:] for l in per_label_ids.values()]))

    def data(self, train=True):
        for album_id, label in self.label_dict.items():

            # We can use only the albums in the training set when we are in training
            if train and album_id not in self.training_albums:
                continue
            if not train and album_id not in self.validation_albums:
                continue

            image_path = os.path.join(self.image_folder, album_id)
            # If path doesn't exist, continue
            if not os.path.exists(image_path):
                continue

            images = [os.path.join(image_path, img_name)
                      for img_name in sorted(os.listdir(image_path))]
            # If no photo available, continue
            if len(images) == 0:
                continue

            yield images, self.label_to_index[label[0]]

