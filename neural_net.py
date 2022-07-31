import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler, TensorDataset, random_split, SubsetRandomSampler, ConcatDataset

import numpy as np
import pandas as pd

class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=3, stride=2),
                                      nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=3, stride=2),
                                      nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=3, stride=2),
                                      nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=3, stride=2),
                                      nn.AdaptiveAvgPool2d((6, 6))
                                     )
        
        self.classifier = nn.Sequential(nn.Linear(256 * 6 * 6, 4096),
                                        nn.Dropout(p=0.5),
                                        nn.Linear(4096, 4096),
                                        nn.Dropout(p=0.5),
                                        nn.Linear(4096, 2)
                                       )


    def forward(self, x):
        x = self.features(x)
        
        x = x.view(-1, 6 * 6 * 256)
        
        x = self.classifier(x)
        return x

def prediction(image):
    PATH = "melanoma_CNN.pt"

    model = torch.load(PATH, map_location=torch.device('cpu'))
    model.eval()

    transform = transforms.Compose([
    transforms.Resize((227, 227)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    classes = ['Benign', 'Malignant']

    batch_t = torch.unsqueeze(transform(image), 0)
    output = model(batch_t)

    ps = torch.exp(output)
    _, pred = torch.max(ps,1)
    prob = torch.nn.functional.softmax(output, dim=1)
    prediction = torch.max(prob, 1)
    return classes[pred[0]], prediction[0].item()*100
