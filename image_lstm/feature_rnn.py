import torch
import torch.nn.functional as F
from image_lstm.embedding_loader import EmbeddingExtractor


class FeatureRNN(torch.nn.Module):

    def __init__(self, size_input, size_hidden, n_labels):
        super(FeatureRNN, self).__init__()
        self.size_hidden = size_hidden
        self.mixed_size = size_input + size_hidden
        self.i2h = torch.nn.Linear(self.mixed_size, size_hidden)
        self.i2o = torch.nn.Linear(self.mixed_size, n_labels)
        self.softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, feature, hidden):
        mixed = torch.cat((feature, hidden), 1)
        new_hidden = self.i2h(mixed)
        labels = self.softmax(self.i2o(mixed))
        return labels, new_hidden

    def init_hidden(self):
        return torch.zeros(1, self.size_hidden)