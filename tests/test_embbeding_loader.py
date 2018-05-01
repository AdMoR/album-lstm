from unittest import TestCase
from image_lstm.embedding_loader import AlexNetFC3


class TestEmbbedingLoader(TestCase):

    def setUp(self):
        self.el = AlexNetFC3()

    def test_embedding_loading(self):
        img = self.el.load_img_from_path("/Users/amorvan/Documents/Databases/moonpig_all/7339.jpg")
        print(img.shape)
        feature = self.el.forward(img)
        print(feature.shape)