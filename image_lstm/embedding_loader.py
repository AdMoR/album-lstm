from torchvision import models
from torch import Tensor
from torch import nn
from skimage import io, transform




class AlexNetFC3(nn.Module):

    def __init__(self):
        super(AlexNetFC3, self).__init__()
        original_model = models.resnet18(pretrained=True)
        feature_map = list(original_model.children())[:-1]
        self.extractor = nn.Sequential(*feature_map)

    def forward(self, x):
        x = self.extractor(x)
        return x

    def load_img_from_path(self, img_path):
        x = io.imread(img_path)
        x = transform.resize(x, (224, 224))
        x = x.swapaxes(0, 2)
        x = Tensor(x)
        x = x.unsqueeze(0)
        return x
