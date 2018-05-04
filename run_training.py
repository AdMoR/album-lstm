import pickle
from tensorboardX import SummaryWriter
import torch
from torch import nn, optim, LongTensor
from image_lstm.feature_lstm import FeatureLSTM
from CUFED_loader.cufed_parse import Cufed


#torch.set_default_tensor_type('torch.cuda.FloatTensor')
loader = Cufed("./CUFED_mini/images", "./CUFED_mini/event_type.json")
lstm = FeatureLSTM(512, len(loader.label_to_index), 100, 100)
loss_function = nn.NLLLoss()
optimizer = optim.Adam(lstm.parameters(), lr=0.05)
writer = SummaryWriter()


for epoch in range(4):
    # Training step
    all_losses = []
    for album, label in loader.data():
        print(album, label)
        label = LongTensor([label])#.cuda

        lstm.zero_grad()
        lstm.init_hidden()
        predicted_label = lstm.forward(album).unsqueeze(0)

        loss = loss_function(predicted_label, label)
        all_losses.append(loss.numpy()[0])
        loss.backward()
        optimizer.step()
    writer.add_scalar("training/loss", sum(all_losses) / len(all_losses), epoch)

    # Validation step
    all_losses = []
    for album, label in loader.data(train=False):
        label = LongTensor([label]).cuda()

        lstm.zero_grad()
        lstm.init_hidden()
        predicted_label = lstm.forward(album).unsqueeze(0)
        loss = loss_function(predicted_label, label)

        all_losses.append(loss.numpy()[0])
    writer.add_scalar("validation/loss", sum(all_losses) / len(all_losses), epoch)

writer.export_scalars_to_json("./training_and_val_scalars.json")
writer.close()

pickle.dump((lstm, loader.label_to_index), open("model.pkl", "wb"))