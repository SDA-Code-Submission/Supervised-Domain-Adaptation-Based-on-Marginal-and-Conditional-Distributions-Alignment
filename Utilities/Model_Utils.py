import torch
import torch.nn as nn
import torchvision
import timm
from Utilities.TLIB_Utils import ImageClassifier
import TLIB.common.vision.models as models


##
def get_model(hp, n_classes, device):
    if hp.Src.startswith('M') or hp.Src.startswith('U'):
        net = DigitsBackbone(device=device, n_classes=10)
    if hp.Src in ['A', 'W', 'D', 'S']:
        backbone = get_sota_model('vgg16', pretrain=True)
        net = ImageClassifier(backbone, num_classes=n_classes, bottleneck_dim=256,
                              pool_layer=None, finetune=True)
        net.backbone = torchvision.models.vgg16(pretrained=True).features
        net.bottleneck[0] = nn.Linear(512, 256)
    if hp.Src.startswith('CityCam'):
        net = CityCamBackbone(device=device, input_dim=2048)
    return net

## Model for Digits tasks
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)


class DigitsBackbone(torch.nn.Module):
    def __init__(self, device, n_classes=10, hidden_dim=1152):
        super(DigitsBackbone, self).__init__()
        self.device = device
        self.phi = nn.Sequential(
            nn.Conv2d(1, 32, 3),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 3),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Dropout(0.5),
            Flatten(),
            nn.Linear(hidden_dim, 120),
            nn.ReLU(True),
            nn.Linear(120, 84),
            nn.ReLU(True)
        )

        self.hypothesis = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(84, n_classes),
        )

    def forward(self, x):
        embedded_x = self.phi(x)
        outputs = self.hypothesis(embedded_x)
        if not self.training:
            return outputs
        else:
            return outputs, embedded_x


##  Model for Office\VisDA-C tasks
def get_sota_model(model_name, pretrain=True):
    """
    Load models from pytorch\timm
    """
    if model_name in models.__dict__:
        # load models from common.vision.models
        backbone = models.__dict__[model_name](pretrained=pretrain)
    else:
        # load models from timm
        backbone = timm.create_model(model_name, pretrained=pretrain)
        try:
            backbone.out_features = backbone.get_classifier().in_features
            backbone.reset_classifier(0, '')
        except:
            backbone.out_features = backbone.head.in_features
            backbone.head = nn.Identity()
    return backbone


## Model for CitCam dataset
class CityCamBackbone(torch.nn.Module):
    def __init__(self, device, input_dim, dof=0.5):
        super(CityCamBackbone, self).__init__()
        self.device = device
        self.dof = dof
        self.phi = nn.Sequential(
            nn.Linear(input_dim, 100),
            nn.ReLU(True),
            nn.Linear(100, 10)
        )
        self.hypothesis = nn.Sequential(
            nn.Dropout(self.dof),
            nn.Linear(10, 1),
        )

    def forward(self, x, ):
        embedded_x = self.phi(x)
        outputs = self.hypothesis(embedded_x)
        if not self.training:
            return outputs
        else:
            return outputs, embedded_x

