import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

#################################################################################
########################### Pretrained Models
#################################################################################
class MODEL(nn.Module):
    def __init__(
        self, model_ft, model_name, num_layer_discard=1, other_features_length=62, num_classes=27,
        text_size=400, text_hidden_size=400, num_embeddings=6000
        ):
        super(MODEL, self).__init__()
        # Image feature extractor
        self.features = nn.Sequential(
            *list(model_ft.children())[:-num_layer_discard]
        )

        # Other feature extractor
        self.fc_1 = nn.Linear(other_features_length, 200)
        self.fc_2 = nn.Linear(200,400)

        # Description feature extractor
        self.hidden_size = text_hidden_size
        self.embedding = nn.Linear(num_embeddings, text_size)
        self.lstm_cell = nn.LSTMCell(text_size, text_hidden_size)
        self.i2o = nn.Linear(text_size + text_hidden_size, text_hidden_size)

        # Final FC
        self.dropout = nn.Dropout(p=0.25)
        self.fc_dropout = nn.Dropout(p=0.5)
        if model_name == 'densenet161':
            image_ft_size = 4416
        elif model_name == 'densenet121':
            image_ft_size = 1024*2*1
        elif model_name == 'vgg19_bn' or model_name == 'vgg16_bn':
            image_ft_size = 512*7*7
        elif model_name == 'mobilenet_v2':
            image_ft_size = 1280*3*2
        else:
            image_ft_size = 2048
        self.fc = nn.Linear(image_ft_size+400+text_hidden_size, num_classes)
        
    def forward(self, x, other_ft, text):
        # Image features
        image_ft = self.features(x)
        image_ft = image_ft.view(x.size(0),-1)
        # Other information features
        other_ft = F.relu(self.fc_1(other_ft))
        other_ft = F.relu(self.fc_2(other_ft))
        # Description features
        text_hidden = self.initHidden(text)
        for i in range(text.size(1)):
            text_ft, text_hidden = self._lstm(text[:,i], text_hidden)
        # Combine features
        all_ft = torch.cat((image_ft, other_ft, text_ft), 1)
        all_ft = self.dropout(all_ft)
        scores = self.fc(all_ft)
        return scores
    
    def _lstm(self, input, hidden_and_cell):
        hidden = hidden_and_cell[0]
        input = self.embedding(input)
        combined = torch.cat((input, hidden), 1)
        hidden_and_cell = self.lstm_cell(input, hidden_and_cell)
        combined = self.fc_dropout(combined)
        output = self.i2o(combined)
        return output, hidden_and_cell
    
    def initHidden(self, input):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batch_size = input.size(0)
        return (torch.zeros(batch_size, self.hidden_size).to(device), torch.zeros(batch_size, self.hidden_size).to(device))

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_pretrained_model(
    model_name, num_classes, other_features_length=62, feature_extract=False, use_pretrained=True,
    text_size=400, text_hidden_size=400, num_embeddings=6000
    ):
    if model_name == 'resnet101':
        """ Resnet101
        """
        model_ft = models.resnet101(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
    elif model_name == 'wide_resnet101_2':
        """ Wide Resnet101_2
        """
        model_ft = models.wide_resnet101_2(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
    elif model_name == 'densenet161':
        """ Densenet 161
        """
        model_ft = models.densenet161(pretrained=True)
        set_parameter_requires_grad(model_ft, feature_extract)
    elif model_name == 'resnet152':
        """ Resnet152
        """
        model_ft = models.resnet152(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
    elif model_name == 'resnext101_32x8d':
        """ ResNeXt-101-32x8d
        """
        model_ft = models.resnext101_32x8d(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
    elif model_name == 'vgg19_bn':
        """ VGG 19 with Batchnorm
        """
        model_ft = models.vgg19_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
    elif model_name == 'resnext50_32x4d':
        """ ResNeXt-50-32x4d
        """
        model_ft = models.resnext50_32x4d(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
    elif model_name == 'mobilenet_v2':
        """ Mobilenet V2
        """
        model_ft = models.mobilenet_v2(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
    elif model_name == 'resnet50':
        """ Resnet50
        """
        model_ft = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
    elif model_name == 'densenet121':
        """ Densenet 121
        """
        model_ft = models.densenet121(pretrained=True)
        set_parameter_requires_grad(model_ft, feature_extract)
    elif model_name == 'vgg16_bn':
        """ VGG 19 with Batchnorm
        """
        model_ft = models.vgg16_bn(pretrained=True)
        set_parameter_requires_grad(model_ft, feature_extract)

    model = MODEL(
        model_ft, model_name, num_layer_discard=1, other_features_length=other_features_length, num_classes=num_classes,
        text_size=text_size, text_hidden_size=text_hidden_size, num_embeddings=num_embeddings
        )
    return model