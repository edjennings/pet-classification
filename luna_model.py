import torchvision.models as models
import torch.nn as nn

#load pre-trained default weights
resnet18 = models.resnet18(weights='DEFAULT')

#freeze parameters in all layers to prevent training
for param in resnet18.parameters():
    param.requires_grad = False


#replace last layer in original model with custom, trainable layer
fc_inputs = resnet18.fc.in_features
resnet18.fc = nn.Sequential(
    nn.Linear(fc_inputs, 512),
    nn.ReLU(),
    nn.Linear(512,3),
    nn.Softmax(dim=0)
)
