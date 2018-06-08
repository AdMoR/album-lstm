import torch
import torch.nn.functional as F
from image_lstm.embedding_loader import EmbeddingExtractor


class FeatureLSTM(torch.nn.Module):

    def __init__(self, feature_size, n_classes, hidden_size):
        super(FeatureLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.feature_size = feature_size

        self.lstm = torch.nn.LSTM(feature_size, self.hidden_size)
        self.hidden_to_tags = torch.nn.Linear(self.hidden_size, n_classes)
        self.hidden = (torch.zeros(1, 1, self.hidden_size),
                       torch.zeros(1, 1, self.hidden_size))
        self.feature_loader = EmbeddingExtractor()

    def init_hidden(self, batch_size):
        self.hidden = (torch.zeros(1, batch_size, self.hidden_size),
                       torch.zeros(1, batch_size, self.hidden_size))

    def forward(self, photos):
        batch = self.build_batch(photos)
        lstm_out, self.hidden = self.lstm(batch, self.hidden)
        print(lstm_out.shape)
        tags = self.hidden_to_tags(lstm_out[-1, :, :])
        tag_scores = F.softmax(tags)
        return tag_scores

    def build_batch(self, photos):
        full_batch = []
        for images_slice in zip(*photos):
            batch_size = len(images_slice)
            batch = torch.cat([self.feature_loader.load_img_from_path(image_path)
                               for image_path in images_slice], 0)
            batch = self.feature_loader.forward(batch).view(batch_size, -1)
            full_batch.append(batch.unsqueeze(0))
        return torch.cat(full_batch, 0)

    def test(self, data):
        correct = 0
        incorrect = 0
        for album, label in  data:
            self.zero_grad()
            self.init_hidden()
            predicted_label = int(torch.argmax(self.forward(album)).numpy())

            if predicted_label == label:
                correct += 1
            else:
                incorrect += 1

        return correct, incorrect



