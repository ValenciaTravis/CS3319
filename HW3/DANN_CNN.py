from preprocess import *
import torch
import torch.nn as nn
import math
from torch.autograd import Function

class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x) # to ensure backward is called
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambd, None


def grad_reverse(x, lambd=1.0):
    return GradReverse.apply(x, lambd)

class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(5, 32, kernel_size=3, padding=1),  # (32,8,9)
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, padding=1), # (64,8,9)
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.MaxPool2d(2),  # (64,4,4)

            nn.Conv2d(64, 128, kernel_size=3, padding=1), # (128,4,4)
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1))  # (128,1,1)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # (batch, 128)
        return x

class LabelClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.fc(x)
    
class DomainClassifier(nn.Module):
    def __init__(self, num_domains=12):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_domains)
        )

    def forward(self, x, lambd):
        x = grad_reverse(x, lambd)
        return self.fc(x)
    
class DANN(nn.Module):
    def __init__(self, num_classes=3, num_domains=12):
        super().__init__()
        self.feature = FeatureExtractor()
        self.class_classifier = LabelClassifier(num_classes)
        self.domain_classifier = DomainClassifier(num_domains)

    def forward(self, x, lambd=1.0):
        feat = self.feature(x)

        class_output = self.class_classifier(feat)
        domain_output = self.domain_classifier(feat, lambd)

        return class_output, domain_output
    
    def get_feature(self, x):
        feat = self.feature(x)
        return feat
