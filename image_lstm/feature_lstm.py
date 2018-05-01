import torch
import torch.nn.functional as F
from image_lstm.embedding_loader import AlexNetFC3


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
        self.feature_loader = AlexNetFC3()

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
        n_batch = int(len(photos) / batch_size) + 1
        for i in range(n_batch):
            imgs = torch.cat([self.feature_loader.load_img_from_path(photo_path)
                              for photo_path in photos[i * batch_size: (i + 1) * batch_size]])
            if feats is None:
                feats = self.feature_loader.forward(imgs)
            else:
                feats = torch.cat([feats, self.feature_loader.forward(imgs)])

        return feats.view(len(photos), self.feature_size)


