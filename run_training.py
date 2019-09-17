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


"""
We train a RNN with feature extracted from a resnet to predict the theme of an album
"""

# 0 - Setup torch env
cuda_enabled = True
if cuda_enabled:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    label_format = lambda label: LongTensor(label).cuda()
else:
    label_format = lambda label: LongTensor(label)

# 1 - setup the dataset
loader = Cufed("./CUFED/images", "./CUFED/event_type.json")
json.dump(loader.label_to_index, open("label_to_index", "w"))
loss_function = nn.NLLLoss()

# We do a random search of the hyperparameters : lr, hidden layer size
for try_ in range(10):

    writer = SummaryWriter()
    # The hidden layer size
    h_size = int(10 ** (2.5 + 1.7 * random.random()))
    # Learning rate
    lr = 10 ** (-2 - 2 * random.random())
    batch_size = 2

    sequence_model = FeatureRNN(512, h_size, len(loader.label_to_index)).cuda()
    optimizer = optim.Adam(list(sequence_model.i2o.parameters()) +\
                           list(sequence_model.i2h.parameters()), lr=lr)

    print("\n\n\n")

    for epoch in range(50):
        # Training step
        all_losses = []
        print("Running with the following parameters >>>", h_size, lr, batch_size)
        for albums, labels in loader.data(batch_size=batch_size):

            current_batch_size = len(albums)
            labels = label_format(labels)

            # 3 - Init the RNN
            sequence_model.zero_grad()
            sequence_model.init_hidden(batch_size=current_batch_size)

            # 4 - forward
            predicted_label = sequence_model.forward(albums)

            # 5 - backward
            loss = loss_function(predicted_label, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm(list(sequence_model.i2o.parameters()) +\
                                          list(sequence_model.i2h.parameters()), 0.1)
            optimizer.step()
            results = torch.argmax(predicted_label, 1) == labels
            all_losses.append(torch.sum(results))

        print("train ", float(sum(all_losses)) / len(all_losses))
        writer.add_scalars("training/loss",
                           {"{}_{}_{}".format(h_size, lr, batch_size): sum(all_losses) / len(all_losses)}, epoch)

        # Validation step
        all_losses = []
        for album, label in loader.data(train=False, batch_size=1):
            label = label_format(label)

            sequence_model.zero_grad()
            sequence_model.init_hidden(batch_size=1)
            predicted_label = sequence_model.forward(album)
            result = torch.argmax(predicted_label, 1) == label
            all_losses.append(result)
        print("val ", float(sum(all_losses)) / len(all_losses))
        writer.add_scalars("validation/loss",
                           {"{}_{}_{}".format(h_size, lr, batch_size): float(sum(all_losses)) / len(all_losses)}, epoch)

    writer.export_scalars_to_json("./training_and_val_scalars_{}.json".format(try_))
    writer.close()

    torch.save(sequence_model.state_dict, "model_{}_{}_{}_{}.torch".format(try_, h_size, lr,
                                                                           batch_size))

