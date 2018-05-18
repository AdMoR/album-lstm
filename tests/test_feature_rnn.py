import torch
from unittest import TestCase
from image_lstm.embedding_loader import EmbeddingExtractor
from image_lstm.feature_rnn import FeatureRNN


class TestFeatureRNN(TestCase):

    def setUp(self):
        self.embedding_loader = EmbeddingExtractor()
        self.model = FeatureRNN(512, 512, 3)

    def test_forward(self):
        feature = self.embedding_loader.forward(
            self.embedding_loader.load_img_from_path(
                "../CUFED_mini/images/0_7138083@N04/5271641595.jpg")
        ).view(1, -1)
        hidden = self.model.init_hidden()

        labels, hidden = self.model.forward(feature, hidden)
        self.assertEqual(labels.size()[1], 3, "Bad number of labels")

    def test_backward(self):
        feature = self.embedding_loader.forward(
            self.embedding_loader.load_img_from_path(
                "../CUFED_mini/images/0_7138083@N04/5271641595.jpg")
        ).view(1, -1)
        hidden = self.model.init_hidden()

        labels, hidden = self.model.forward(feature, hidden)

        loss_function = torch.nn.NLLLoss()
        loss = loss_function(labels, torch.LongTensor(1))
        loss.backward()