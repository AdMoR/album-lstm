import pickle
from image_lstm.feature_lstm import FeatureLSTM
from torch import nn, optim

lstm = FeatureLSTM(512, 2, 100, 100)
loss_function = nn.NLLLoss()
optimizer = optim.SGD(lstm.parameters(), lr=0.1)
albums = []

for epoch in range(10):
    for album, label in albums:
        batch = lstm.build_batch(album)
        predicted_label = lstm.forward().unsqueeze(0)

        loss = loss_function(predicted_label, label)
        loss.backward()
        optimizer.step()

pickle.dumps(lstm)