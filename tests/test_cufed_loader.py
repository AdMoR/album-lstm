from unittest import TestCase
from CUFED_loader.cufed_parse import Cufed


class TestCufedLoader(TestCase):

    def setUp(self):
        self.loader = Cufed("../CUFED_mini/images", "../CUFED_mini/event_type.json")

    def test_Stats(self):
        print(self.loader.class_stats())

    def test_data_retrieval(self):
        for images, label in self.loader.data():
            print(images, label)
            return