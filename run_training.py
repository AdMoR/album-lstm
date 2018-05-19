import pickle
from tensorboardX import SummaryWriter
import torch
import copy
from torch import nn, optim, LongTensor
from image_lstm.feature_lstm import FeatureLSTM
from image_lstm.feature_rnn import FeatureRNN
from CUFED_loader.cufed_parse import Cufed


cuda_enabled = False

if cuda_enabled:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    label_format = lambda label: LongTensor([label]).cuda
else:
    label_format = lambda label: LongTensor([label])

loader = Cufed("./CUFED_mini/images", "./CUFED_mini/event_type.json")
#sequence_model = FeatureLSTM(512, len(loader.label_to_index), 100, 100)
sequence_model = FeatureRNN(512, 512, len(loader.label_to_index))
loss_function = nn.NLLLoss()
optimizer = optim.Adam(sequence_model.parameters(), lr=0.05)
writer = SummaryWriter()


for epoch in range(4):
    # Training step
    all_losses = []
    for album, label in loader.data():
        print(album, label)
        label = label_format(label)

        sequence_model.zero_grad()
        sequence_model.init_hidden()
        predicted_label = sequence_model.forward(album)

        loss = loss_function(predicted_label, label)
        all_losses.append(loss.item())
        loss.backward()
        optimizer.step()
    writer.add_scalars("training/loss", {"nll": sum(all_losses) / len(all_losses)}, epoch)

    # Validation step
    all_losses = []
    for album, label in loader.data(train=False):
        label = label_format(label)

        sequence_model.zero_grad()
        sequence_model.init_hidden()
        predicted_label = sequence_model.forward(album)
        loss = loss_function(predicted_label, label)

        all_losses.append(loss.item())
    writer.add_scalars("validation/loss", {"nll": sum(all_losses) / len(all_losses)}, epoch)

writer.export_scalars_to_json("./training_and_val_scalars.json")
writer.close()

pickle.dump((sequence_model, loader.label_to_index), open("model.pkl", "wb"))