from unittest import TestCase
from image_lstm.embedding_loader import EmbeddingExtractor


class TestEmbbedingLoader(TestCase):

    def setUp(self):
        self.el = EmbeddingExtractor()

    def test_embedding_loading(self):
        img = self.el.load_img_from_path("/Users/amorvan/Documents/Databases/moonpig_all/7339.jpg")
        print(img.shape)
        feature = self.el.forward(img)
        print(feature.shape)