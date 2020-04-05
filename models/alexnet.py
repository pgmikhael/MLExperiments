import torch
import torch.nn as nn
from models.model_factory import RegisterModel

<<<<<<< HEAD

@RegisterModel("vanilla_alexnet")
class AlexNet(nn.Module):
    def __init__(self, args):
        super(AlexNet, self).__init__()
=======
@RegisterModel("vanilla_alexnet")
class Vanilla_AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(Vanilla_AlexNet, self).__init__()
>>>>>>> 348851778bfd433380a114b4645bd0be16077803
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, args.num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


@RegisterModel("bn_alexnet")
class BN_AlexNet(nn.Module):
    def __init__(self, args):
        super(BN_AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, args.num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

@RegisterModel("supp_alexnet")
class AlexNet_Norm(nn.Module):
    def __init__(self, args):
        super(AlexNet_Norm, self).__init__()
        self.clip_supp_weights = args.clip_supp_weights
        self.features = nn.Sequential(
            Suppressive_Norm((3,64,11,4,2), clip_supp_weights= self.clip_supp_weights),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            Suppressive_Norm((64,192,5,1,2), clip_supp_weights= self.clip_supp_weights),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            Suppressive_Norm((192,384,3,1,1), clip_supp_weights= self.clip_supp_weights),
            nn.ReLU(inplace=True),
            Suppressive_Norm((384,256,3,1,1), clip_supp_weights= self.clip_supp_weights),
            nn.ReLU(inplace=True),
            Suppressive_Norm((256,256,3,1,1), clip_supp_weights= self.clip_supp_weights),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, args.num_classes),
        )

    def forward(self, x):
        for m in self.modules():
            if isinstance(m, Suppressive_Norm):
                torch.clamp_min_(m.supp_conv.weight, 0)

        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class Suppressive_Norm(nn.Module):
    def __init__(self, dim , alpha = 6, clip_supp_weights = False):
        super(Suppressive_Norm, self).__init__()
        self.dim = dim
        self.alpha = alpha
        self.clip_supp_weights = clip_supp_weights
        self.conv = lambda i,o,k,s,p: nn.Conv2d(i, o, kernel_size=k, stride=s, padding=p)
        self.supp_conv = lambda i,o,k,s,p: nn.Conv2d(i, o, kernel_size=k, stride=s, padding=p)
    
    def forward(self, x):
        """
        inc of kernel size by alpha ==> increase of padding by alpha/2 to maintain size
        """

        i,o,k,s,p = self.dim
        if self.clip_supp_weights:
            x_supp = self.conv(i,o,k+self.alpha,s,p + self.alpha//2)(x**2)
            x = self.conv(i,o,k,s,p)(x)
            x = x/(1+torch.sqrt(x_supp))
        else:
            x_supp = self.conv(i,o,k+self.alpha,s,p + self.alpha//2)(x)
            x = self.conv(i,o,k,s,p)(x)
            x = x/(1+x_supp)
        return x