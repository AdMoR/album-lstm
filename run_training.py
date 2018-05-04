import pickle
from image_lstm.feature_lstm import FeatureLSTM
from torch import nn, optim, LongTensor
from CUFED_loader.cufed_parse import Cufed

loader = Cufed("./CUFED/images", "./CUFED/event_type.json")
lstm = FeatureLSTM(512, len(loader.label_to_index), 100, 100)
loss_function = nn.NLLLoss()
optimizer = optim.SGD(lstm.parameters(), lr=0.1)


for epoch in range(1):
    for album, label in loader.data():
        label = LongTensor([label])

        lstm.zero_grad()
        lstm.init_hidden()
        predicted_label = lstm.forward(album).unsqueeze(0)

        loss = loss_function(predicted_label, label)
        loss.backward()
        optimizer.step()

pickle.dumps(lstm)