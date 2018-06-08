import pickle
import json
from tensorboardX import SummaryWriter
import torch
import copy
from torch import nn, optim, LongTensor
from image_lstm.feature_lstm import FeatureLSTM
from image_lstm.feature_rnn import FeatureRNN
from CUFED_loader.cufed_parse import Cufed
import random




cuda_enabled = False

if cuda_enabled:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    label_format = lambda label: LongTensor(label).cuda()
else:
    label_format = lambda label: LongTensor(label)

loader = Cufed("./CUFED_mini/images", "./CUFED_mini/event_type.json")
#sequence_model = FeatureLSTM(512, len(loader.label_to_index), 100, 100)
loss_function = nn.NLLLoss()
writer = SummaryWriter()


for try_ in range(10):

    h_size = int(10 ** (1 + 2 * random.random()))
    lr = 10 ** (-1 - 2 * random.random())
    batch_size = random.randint(2, 4)

    sequence_model = FeatureRNN(512, h_size, len(loader.label_to_index))
    optimizer = optim.Adam(sequence_model.parameters(), lr=lr)

    for epoch in range(20):
        # Training step
        for albums, labels in loader.data(batch_size=batch_size):
            print(albums, labels)
            labels = label_format(labels)

            sequence_model.zero_grad()
            sequence_model.init_hidden(batch_size=batch_size)
            predicted_label = sequence_model.forward(albums)

            loss = loss_function(predicted_label, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm(sequence_model.parameters(), 0.1)
            optimizer.step()

        # Validation step
        all_losses = []
        for album, label in loader.data(train=False):
            label = label_format(label)

            sequence_model.zero_grad()
            sequence_model.init_hidden()
            predicted_label = sequence_model.forward(album)
            loss = loss_function(predicted_label, label)

            all_losses.append(loss.item())
        #writer.add_scalars("validation/loss", {"nll": sum(all_losses) / len(all_losses)}, epoch)

    writer.export_scalars_to_json("./training_and_val_scalars.json")
    writer.close()

    torch.save(sequence_model.state_dict, "model_{}_{}_{}.torch".format(h_size, lr, batch_size))
    json.dump(loader.label_to_index, open("label_to_index", "w"))
