import torch
import torch.nn.functional as F
from image_lstm.embedding_loader import EmbeddingExtractor


class FeatureLSTM(torch.nn.Module):

    def __init__(self, feature_size, n_classes, embedding_size, hidden_size):
        super(FeatureLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.feature_size = feature_size
        self.embedding_size = embedding_size

        self.embedding = torch.nn.Linear(feature_size, embedding_size)
        self.lstm = torch.nn.LSTM(embedding_size, self.hidden_size)
        self.hidden_to_tags = torch.nn.Linear(self.hidden_size, n_classes)
        self.hidden = (torch.zeros(1, 1, self.hidden_size),
                       torch.zeros(1, 1, self.hidden_size))
        self.feature_loader = EmbeddingExtractor()

    def init_hidden(self):
        self.hidden = (torch.zeros(1, 1, self.hidden_size),
                       torch.zeros(1, 1, self.hidden_size))

    def forward(self, photos):
        batch = self.build_batch(photos)
        embeds = self.get_embedding(batch)
        lstm_out, self.hidden = self.lstm(
            embeds.view(len(photos), 1, -1), self.hidden)
        tags = self.hidden_to_tags(lstm_out.view(len(photos), -1))
        tag_scores = F.softmax(tags)
        return tag_scores[-1]

    def get_embedding(self, features):
        return self.embedding(features)

    def build_batch(self, photos):
        feats = None
        batch_size = 50
        for i in range(0, len(photos), batch_size):
            imgs = torch.cat([self.feature_loader.load_img_from_path(photo_path)
                              for photo_path in photos[i: i + batch_size]])
            if feats is None:
                feats = self.feature_loader.forward(imgs)
            else:
                feats = torch.cat([feats, self.feature_loader.forward(imgs)])

        return feats.view(len(photos), self.feature_size)

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



