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


cuda_enabled = True
if cuda_enabled:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    label_format = lambda label: LongTensor(label).cuda()
else:
    label_format = lambda label: LongTensor(label)

loader = Cufed("./CUFED/images", "./CUFED/event_type.json")
json.dump(loader.label_to_index, open("label_to_index", "w"))
#sequence_model = FeatureLSTM(512, len(loader.label_to_index), 100, 100)
loss_function = nn.NLLLoss()


for try_ in range(10):

    writer = SummaryWriter()

    h_size = int(10 ** (1 + 2 * random.random()))
    lr = 10 ** (-1 - 2 * random.random())
    batch_size = 2

    sequence_model = FeatureRNN(512, h_size, len(loader.label_to_index)).cuda()
    optimizer = optim.Adam(list(sequence_model.i2o.parameters()) +\
                           list(sequence_model.i2h.parameters()), lr=lr)

    for epoch in range(20):
        # Training step
        for albums, labels in loader.data(batch_size=batch_size):
            print(len(albums[0]), batch_size, labels)
            current_batch_size = len(albums)
            labels = label_format(labels)

            sequence_model.zero_grad()
            sequence_model.init_hidden(batch_size=current_batch_size)
            predicted_label = sequence_model.forward(albums)

            loss = loss_function(predicted_label, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm(list(sequence_model.i2o.parameters()) +\
                                          list(sequence_model.i2h.parameters()), 0.1)
            optimizer.step()
            print(torch.norm(sequence_model.i2h.weight, 2))

        # Validation step
        all_losses = []
        for album, label in loader.data(train=False, batch_size=1):
            label = label_format(label)

            sequence_model.zero_grad()
            sequence_model.init_hidden(batch_size=1)
            predicted_label = sequence_model.forward(album)
            result = torch.argmax(predicted_label, 1) == label
            print(result)
            all_losses.append(result)
        writer.add_scalars("validation/loss",
                           {"{}_{}_{}".format(h_size, lr, batch_size): sum(all_losses) / len(all_losses)}, epoch)

    writer.export_scalars_to_json("./training_and_val_scalars_{}.json".format(try_))
    writer.close()

    torch.save(sequence_model.state_dict, "model_{}_{}_{}_{}.torch".format(try_, h_size, lr,
                                                                           batch_size))

