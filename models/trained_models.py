from models.model_factory import RegisterModel
import torchvision.models as models
import torch.nn as nn
import torch

@RegisterModel('resnet18')
class Resnet18(nn.Module):
    def __init__(self, args):
        super(Resnet18, self).__init__()
        self._model = models.resnet18(pretrained= args.trained_on_imagenet)
        self._model.fc = nn.Linear(self._model.fc.in_features, args.num_classes)

    def forward(self, x, batch=None):
        return self._model(x)
    
    def forward_thru_convs(self, x, batch = None):
        dummy_model = self._model
        dummy_model.fc = torch.nn.Identity() 
        return dummy_model(x)

@RegisterModel('alexnet')
class AlexNet(nn.Module):
    def __init__(self, args):
        super(AlexNet, self).__init__()
        self._model = models.alexnet(pretrained=args.trained_on_imagenet)
        self._model.classifier = nn.Linear(self._model.classifier.in_features, args.num_classes)

    def forward(self, x, batch=None):
        return self._model(x)

@RegisterModel('vgg16')
class VGG16(nn.Module):
    def __init__(self, args):
        super(VGG16, self).__init__()
        self._model = models.vgg16(pretrained=args.trained_on_imagenet)
        first_layer_output, _ = self._model.classifier[0].out_features 
        _, final_layer_input = self._model.classifier[-1].in_features

        self._model.classifier[0] = nn.Linear(args.rolled_size, first_layer_output)
        self._model.classifier[-1] = nn.Linear(final_layer_input, args.num_classes)

    def forward(self, x, batch=None):
        return self._model(x)

@RegisterModel('densenet161')
class DenseNet161(nn.Module):
    def __init__(self, args):
        super(DenseNet161, self).__init__()
        self._model = models.densenet161(pretrained=args.trained_on_imagenet)
        self._model.classifier = nn.Linear(args.rolled_size, args.num_classes)

    def forward(self, x, batch=None):
        return self._model(x)

# TODO: support for inception not implemented
@RegisterModel('inception_v3')
class Inception_v3(nn.Module):
    def __init__(self, args):
        super(Inception_v3, self).__init__()
        self._model = models.inception_v3(pretrained=args.trained_on_imagenet)
        self._model.AuxLogits.fc = nn.Linear(args.rolled_size, args.num_classes)
        self._model.fc = nn.Linear(args.rolled_size, args.num_classes)

    def forward(self, x, batch=None):
        return self._model(x)



