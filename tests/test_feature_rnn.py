import torch
from unittest import TestCase
from image_lstm.embedding_loader import EmbeddingExtractor
from image_lstm.feature_rnn import FeatureRNN


class TestFeatureRNN(TestCase):

    @classmethod
    def setUpClass(self):
        self.embedding_loader = EmbeddingExtractor()
        self.model = FeatureRNN(512, 512, 3)

    def test_forward(self):
        self.model.init_hidden(batch_size=1)
        labels = self.model.forward([["../CUFED_mini/images/0_7138083@N04/5271641595.jpg",
                                     "../CUFED_mini/images/0_7138083@N04/5271641595.jpg"]])
        print(labels.shape)
        self.assertEqual(labels.size()[1], 3, "Bad number of labels")

    def test_backward(self):
        self.model.init_hidden(batch_size=2)
        labels = self.model.forward([["../CUFED_mini/images/0_7138083@N04/5271641595.jpg"],
                                     ["../CUFED_mini/images/0_7138083@N04/5271641595.jpg"]])
        print(labels.shape)
        loss_function = torch.nn.NLLLoss()
        loss = loss_function(labels, torch.LongTensor([1, 1]))
        loss.backward()