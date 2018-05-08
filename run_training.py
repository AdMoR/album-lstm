import pickle
import json
from tensorboardX import SummaryWriter
import torch
from torch import nn, optim, LongTensor
from image_lstm.feature_lstm import FeatureLSTM
from CUFED_loader.cufed_parse import Cufed


torch.set_default_tensor_type('torch.cuda.FloatTensor')
loader = Cufed("./CUFED/images", "./CUFED/event_type.json")
lstm = FeatureLSTM(512, len(loader.label_to_index), 100, 100).cuda()
loss_function = nn.NLLLoss()
optimizer = optim.Adam(lstm.parameters(), lr=0.005)
writer = SummaryWriter()


for epoch in range(20):
    # Training step
    all_losses = []
    for album, label in loader.data():
        print(album, label)
        label = LongTensor([label]).cuda()

        lstm.zero_grad()
        lstm.init_hidden()
        predicted_label = lstm.forward(album).unsqueeze(0)

        loss = loss_function(predicted_label, label)
        loss.backward()
        optimizer.step()

    # Validation step
    correct, incorrect = lstm.test(loader.data(train=False))
    writer.add_scalars("validation/score", {"correct": correct / (correct + incorrect)}, epoch)

writer.export_scalars_to_json("./training_and_val_scalars.json")
writer.close()

torch.save(lstm.state_dict, "model.torch")
json.dump(loader.label_to_index, open("label_to_index", "w"))
