from unittest import TestCase
from torch import nn, optim
import torch
from image_lstm.embedding_loader import AlexNetFC3
from image_lstm.feature_lstm import FeatureLSTM


class TestFeatureLSTM(TestCase):

    def setUp(self):
        self.lstm = FeatureLSTM(512, 2, 100, 100)
        self.embedding_loader = AlexNetFC3()

        self.album_photos = ["/Users/amorvan/Documents/Databases/moonpig_all/7339.jpg",
                             "/Users/amorvan/Documents/Databases/moonpig_all/7339.jpg",
                             "/Users/amorvan/Documents/Databases/moonpig_all/7339.jpg"]
        self.album_label = torch.LongTensor([1])

    def test_lstm_forward(self):
        tags = self.lstm.forward(self.album_photos)
        print(tags)

    def test_lstm_backward(self):
        loss_function = nn.NLLLoss()
        optimizer = optim.SGD(self.lstm.parameters(), lr=0.1)

        label = self.lstm.forward(self.album_photos).unsqueeze(0)
        print(label)
        old_label = label

        loss = loss_function(label, self.album_label)
        loss.backward()
        optimizer.step()
        new_label = self.lstm.forward(self.album_photos).unsqueeze(0)
        print(new_label)
        self.assertGreaterEqual(new_label[0][1], old_label[0][1])