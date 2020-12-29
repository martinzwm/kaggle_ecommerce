import numpy as np
import pandas as pd
import os
import math
from PIL import Image
import pickle
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, random_split
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import *
from torchvision import transforms, models

# Dependencies
from utilities import *
from dataset import *
from model import * 

# Train
def main(model_name='resnet101'):
    #################################################################################
    ########################### Parameters
    #################################################################################
    num_classes = 27
    num_epochs = 50
    input_size = (80, 60)
    other_features_length = 62
    text_size = 800
    text_hidden_size = 800

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == torch.device('cuda'):
        batch_size = 256
    else:
        batch_size = 32
    
    with open('cleaned_descrips.pickle', 'rb') as f:
        cleaned_descrips = pickle.load(f)
    
    lang = Lang()
    lang.addDataset(cleaned_descrips['train'])
    lang.addDataset(cleaned_descrips['val'])
    lang.addDataset(cleaned_descrips['test'])

    #################################################################################
    ########################### Pretrained Models
    #################################################################################
    feature_extract = False # we want to fine tune
    # Initialize the model for this run
    model = initialize_pretrained_model(
        model_name, num_classes, other_features_length, feature_extract, use_pretrained=True,
        text_size=text_size, text_hidden_size=text_hidden_size, num_embeddings=lang.n_words
        )
    model = model.to(device)

    params_to_update = model.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name,param in model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t",name)
    else:
        for name,param in model.named_parameters():
            if param.requires_grad == True:
                print("\t",name)

    # Print the model we just instantiated
    print(model)

    #################################################################################
    ########################### Data Loader
    #################################################################################
    data_transforms = data_aug(input_size)
    train_set = ImageDataset(mode='train', transform=data_transforms['train'], lang=lang)
    val_set = ImageDataset(
        mode='val', transform=data_transforms['val'], cat_to_label=train_set.cat_to_label, 
        label_to_cat=train_set.label_to_cat, gen_to_vec=train_set.gen_to_vec, color_to_vec=train_set.color_to_vec,
        season_to_vec=train_set.season_to_vec, usage_to_vec=train_set.usage_to_vec,
        lang=lang)
    
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        num_workers=0,
        shuffle=True
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        num_workers=1,
        shuffle=True
    )

    #################################################################################
    ########################### Start training
    #################################################################################
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_loss_list, train_acc_list, val_acc_list = train(
        model, model_name, train_loader, val_loader, loss_fn, optimizer, num_epochs, print_every=10, device=device)

    # Save train loss and val loss profile
    train_loss_profile = {
        'train_loss': np.array(train_loss_list),
        'train_acc': np.array(train_acc_list),
        'val_acc': np.array(val_acc_list)
    }
    df = pd.DataFrame.from_dict(train_loss_profile)
    df.to_pickle('train_profile.pickle')

# Validation
def ensemble_val(model_names, debug=False, use_heuristics=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == torch.device('cuda'):
        batch_size = 256
    else:
        batch_size = 32
    
    input_size = (80,60)
    data_transforms = data_aug(input_size)

    with open('cleaned_descrips.pickle', 'rb') as f:
        cleaned_descrips = pickle.load(f)
    
    lang = Lang()
    lang.addDataset(cleaned_descrips['train'])
    lang.addDataset(cleaned_descrips['val'])
    lang.addDataset(cleaned_descrips['test'])

    train_set = ImageDataset(mode='train', transform=data_transforms['train'], lang=lang)
    val_set = ImageDataset(
        mode='val', transform=data_transforms['val'], cat_to_label=train_set.cat_to_label, 
        label_to_cat=train_set.label_to_cat, gen_to_vec=train_set.gen_to_vec, color_to_vec=train_set.color_to_vec,
        season_to_vec=train_set.season_to_vec, usage_to_vec=train_set.usage_to_vec, lang=lang)
    
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        num_workers=1,
    )

    # Get predictions
    preds_all = torch.empty(len(val_set),1).to(device)
    for model_name in model_names:
        model = torch.load('best_model_{}.pt'.format(model_name)).to(device)
        val_acc, _, preds = check_accuracy(model, val_loader, device=device, debug=True)
        print('{} has accuracy {}'.format(model_name, 100*val_acc))
        preds = preds.view(-1,1)
        preds_all = torch.cat((preds_all, preds), dim=1)
    
    preds_ensemble, counts_ensemble = torch.mode(preds_all, dim=1)

    if use_heuristics:
        preds_ensemble = ensemble_heuristics(
            preds_ensemble, counts_ensemble, N=len(model_names), texts=cleaned_descrips['val']
            )

    # Get labels
    labels = torch.zeros(len(val_set)).to(device)
    for i in range(len(val_set)):
        labels[i] = val_set[i]['label']

    num_correct = (preds_ensemble == labels).sum()
    ensemble_acc = 100.0*num_correct/len(labels)
    print('Ensemble has accuracy {}'.format(ensemble_acc))

    # Save predictions to pickle file
    preds_dict = {}
    for i in range(len(model_names)):
        model_name = model_names[i]
        preds_model = preds_all[:,i].cpu().numpy().astype('int32')
        preds_dict[model_name] = preds_model
    preds_dict['gt'] = labels.cpu().numpy().astype('int32')
    preds_dict = {
        'preds_dict': [preds_dict]
    }
    df = pd.DataFrame.from_dict(preds_dict)
    df.to_pickle('preds_dict.pickle')

    if debug: # we want to see which categories are the mistakes being made
        mistake_index = preds_ensemble!=labels
        mistakes = torch.cat((labels[mistake_index].view(-1,1), preds_ensemble[mistake_index].view(-1,1)), dim=1)
        val_mistakes = val_set.label_to_cat[mistakes.cpu().numpy().astype('int32')] # convert mistake labels to categories

        val_mistakes = {
            'val_mistakes': [val_mistakes]
        }
        df = pd.DataFrame.from_dict(val_mistakes)
        df.to_pickle('val_mistakes.pickle')
    return ensemble_acc

# Test
def ensemble_test(model_names):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == torch.device('cuda'):
        batch_size = 256
    else:
        batch_size = 32

    input_size = (80,60)
    data_transforms = data_aug(input_size)

    with open('cleaned_descrips.pickle', 'rb') as f:
        cleaned_descrips = pickle.load(f)
    
    lang = Lang()
    lang.addDataset(cleaned_descrips['train'])
    lang.addDataset(cleaned_descrips['val'])
    lang.addDataset(cleaned_descrips['test'])

    train_set = ImageDataset(mode='train', transform=data_transforms['train'], lang=lang)
    test_set = ImageDataset(
        mode='test', transform=data_transforms['val'], cat_to_label=train_set.cat_to_label, 
        label_to_cat=train_set.label_to_cat, gen_to_vec=train_set.gen_to_vec, color_to_vec=train_set.color_to_vec,
        season_to_vec=train_set.season_to_vec, usage_to_vec=train_set.usage_to_vec, lang=lang)

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        num_workers=1,
    )

    # Get predictions
    preds_all = torch.empty(len(test_set),1).to(device)
    for model_name in model_names:
        model = torch.load('best_model_{}.pt'.format(model_name)).to(device)
        preds = test_single(model, test_loader, device=device)
        preds = preds.view(-1,1)
        preds_all = torch.cat((preds_all, preds), dim=1)
    preds_ensemble = torch.mode(preds_all, dim=1)[0]

    preds_category = test_set.label_to_cat[preds_ensemble.cpu().numpy().astype('int32')]

    preds_category_df = pd.DataFrame(preds_category)
    preds_category_df.to_csv('preds_category.csv', index=False)

if __name__ == '__main__':
    acc = ensemble_val(['resnet50', 'resnet152', 'resnext101_32x8d', 'densenet161', 'wide_resnet101_2', 'densenet121', 'vgg16_bn', 'vgg19_bn'], False, False)