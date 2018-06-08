import torch
import torch.nn.functional as F
from image_lstm.embedding_loader import EmbeddingExtractor


class FeatureRNN(torch.nn.Module):

    def __init__(self, size_input, size_hidden, n_labels):
        super(FeatureRNN, self).__init__()

        # Embbedding loader
        self.embedding_loader = EmbeddingExtractor()

        # RNN params
        self.size_hidden = size_hidden
        self.mixed_size = size_input + size_hidden
        self.i2h = torch.nn.Linear(self.mixed_size, size_hidden)
        self.i2o = torch.nn.Linear(self.mixed_size, n_labels)
        self.softmax = torch.nn.LogSoftmax(dim=1)
        self.init_hidden(batch_size=1)

    def forward(self, photos):
        if type(photos) != list:
            photos = [[photos]]
        features = self.build_batch(photos)

        for feature in features:
            mixed = torch.cat((feature, self.hidden), 1)
            self.hidden = self.i2h(mixed)

        labels = self.softmax(self.i2o(mixed))
        return labels

    def init_hidden(self, batch_size):
        self.hidden = torch.zeros(batch_size, self.size_hidden)

    def build_batch(self, photos):
        full_batch = []
        for images_slice in zip(*photos):
            batch_size = len(images_slice)
            batch = torch.cat([self.embedding_loader.load_img_from_path(image_path)
                               for image_path in images_slice], 0)
            batch = self.embedding_loader.forward(batch).view(batch_size, -1)
            full_batch.append(batch)
        return full_batch

    def old_build_batch(self, photos):
        return [self.embedding_loader.forward(
            self.embedding_loader.load_img_from_path( img_path)
        ).view(1, -1)
        for img_path in photos]