import json
import os
import random
import itertools
from collections import defaultdict


class Cufed(object):
    """
    CUFED is a datset of albums with a label (like "Urban Trip") for each album.

    """

    def __init__(self, image_folder, label_path, reduce_label=False):
        self.image_folder = image_folder
        self.label_dict = json.load(open(label_path))
        self.prepare_classes_to_index()

        self.label_reduction =\
            {"Important Personal Event": ["Wedding", "Birthday", "Graduation"],
             "Personal Activity": ["Personal Sports"],
             "Personal Trip": ["Urban Trip", "Cruise", "Theme Park"],
             "Holiday": ["Christmas", "Halloween"]}

    ######################
    #   MAIN FUNCTIONS   #
    ######################

    def data(self, train=True, batch_size=2):
        """
        We collect a dataset D=[(x, y), ...]
        Where x has the format [n_batch, n_images_in_album, feature_in_image] and y is an index

        We sort D based on the number of images in the album of x to improve batch efficency

        We call 
        1. prepare_batch
        2. batchify
        3. yield result
        """
        if train:
            elements = self.prepare_batch(self.training_albums)
        else:
            elements = self.prepare_batch(self.validation_albums)

        while len(elements) > 0:
            # Collect the batch
            batch = []
            for _ in range(min(batch_size, len(elements))):
                batch.append(elements.pop())

            # Get same sequence size for all elements of the batch
            albums, labels = self.batchify(batch)
            yield  albums, labels

    def prepare_batch(self, iterator):
        """
        Elements are ordered to avoid having to sample too much. 
        """
        elements = []

        for label, album_ids in iterator:
            for album_id in album_ids:
                image_path = os.path.join(self.image_folder, album_id)
                # If path doesn't exist, continue
                if not os.path.exists(image_path):
                    continue
                images = [os.path.join(image_path, img_name)
                          for img_name in sorted(os.listdir(image_path))]
                # If no photo available, continue
                if len(images) == 0:
                    continue

                elements.append((label, images))

        random.shuffle(elements)

        return sorted(elements, key=lambda p: len(p[1]), reverse=True)

    def batchify(self, batch, n_images=None):
        """
        We form batches for the RNN, the expected format is 
        [n_batch, n_images_in_album, feature_in_image]

        Here we can subselect the number of images per album to reduce memory consumption
        """
        # We have two choices take the min number of the two albums or just a number inferior to it
        min_nb_elems = min([len(pair[1]) for pair in batch])
        if n_images is not None:
            min_nb_elems = min(n_images, min_nb_elems)
        # Get a sorted list of indexes to sample from the image list that we have for each album
        selected_elements = [sorted(random.sample(list(range(len(pair[1]))), k=min_nb_elems))
                             for pair in batch]
        # For each element from the batch, we subsample to have the same size everywhere
        return [[path
                 for i, path in enumerate(pair[1])
                 if i in selected_elements[k]]
                for k, pair in enumerate(batch)], \
               [pair[0] for pair in batch]

    #######################
    #   Various helpers.  #
    #######################

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

        self.training_albums = []
        self.training_albums += [(k, v[:-3]) for k, v in per_label_ids.items()]
        self.validation_albums = []
        self.validation_albums+= [(k, v[-3:]) for k, v in per_label_ids.items()]
        #list(itertools.chain(*[(k, v[-3:]) for k, v in per_label_ids.items()]))
