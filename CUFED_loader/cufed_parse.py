import json
import os
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
        for k, v in self.label_dict.items():
            if type(v) == list:
                v = v[0]
            if v not in self.label_to_index.keys():
                self.label_to_index[v] = label_index
                label_index += 1

    def data(self):
        for album_id, label in self.label_dict.items():
            image_path = os.path.join(self.image_folder, album_id)
            if not os.path.exists(image_path):
                continue
            images = [os.path.join(image_path, img_name) for img_name in sorted(os.listdir(image_path))]
            yield images, self.label_to_index[label[0]]

