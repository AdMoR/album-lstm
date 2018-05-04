import pickle
from image_lstm.feature_lstm import FeatureLSTM
import torch
from torch import nn, optim, LongTensor
from CUFED_loader.cufed_parse import Cufed

torch.set_default_tensor_type('torch.cuda.FloatTensor')
loader = Cufed("./CUFED_mini/images", "./CUFED_mini/event_type.json")
lstm = FeatureLSTM(512, len(loader.label_to_index), 100, 100)
loss_function = nn.NLLLoss()
optimizer = optim.Adam(lstm.parameters(), lr=0.05)


for epoch in range(4):
    for album, label in loader.data():
        print(album, label)
        label = LongTensor([label]).cuda()

        lstm.zero_grad()
        lstm.init_hidden()
        predicted_label = lstm.forward(album).unsqueeze(0)

        loss = loss_function(predicted_label, label)
        loss.backward()
        optimizer.step()

pickle.dump((lstm, loader.label_to_index), open("model.pkl", "wb"))