from unittest import TestCase
from torch import nn, optim
import torch
import os
from image_lstm.embedding_loader import EmbeddingExtractor
from image_lstm.feature_lstm import FeatureLSTM


class TestFeatureLSTM(TestCase):

    def setUp(self):
        self.lstm = FeatureLSTM(512, 3, 100)
        self.embedding_loader = EmbeddingExtractor()

        image_dir = "../CUFED_mini/images/0_7138083@N04"
        album = [os.path.join(image_dir, f) for f in os.listdir(image_dir)]
        self.album_photos = [album[:10], album[10:20]]
        self.album_label = torch.LongTensor([0, 1])

    def test_lstm_forward(self):
        self.lstm.init_hidden(batch_size=2)
        tags = self.lstm.forward(self.album_photos)
        print(tags.shape)

    def test_lstm_backward(self):
        loss_function = nn.NLLLoss()
        optimizer = optim.SGD(self.lstm.lstm.parameters(), lr=0.1)
        self.lstm.init_hidden(batch_size=2)
        label = self.lstm.forward(self.album_photos)
        print(label, self.album_label.shape)

        loss = loss_function(label, self.album_label)
        loss.backward()
        optimizer.step()
