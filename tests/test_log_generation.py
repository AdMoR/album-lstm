from unittest import TestCase
from tensorboardX import SummaryWriter
import torch
import os


class TestStatWriter(TestCase):

    def setUp(self):
        self.writer = SummaryWriter()

    def test_writing(self):
        self.writer.add_scalars("test/loss", {"base": torch.rand(1)}, 1)
        self.writer.add_scalars("test/loss", {"base": torch.rand(1)}, 2)
        self.writer.add_scalars("test/loss", {"base": torch.rand(1)}, 3)
        self.writer.export_scalars_to_json("./test.json")
        self.writer.close()
        #os.remove("./test.json")
