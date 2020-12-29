from torch.utils.data import Dataset, WeightedRandomSampler
import numpy as np
import pandas as pd
import os
import pickle
import torch
from PIL import Image
from torchvision import transforms

#################################################################################
########################### Data Loader
#################################################################################
class ImageDataset(Dataset):
    def __init__(
        self, root_dir='uw_cs480_fall20', mode='train', transform=None, 
        cat_to_label=None, label_to_cat=None, gen_to_vec=None,
        color_to_vec=None, season_to_vec=None, usage_to_vec=None,
        cleaned_descrips='cleaned_descrips.pickle', lang=None
        ):
        self.root_dir = root_dir
        self.transform = transform
        if mode == 'train':
            self.data = pd.read_csv(os.path.join(root_dir, 'train.csv'))
            self.cat_to_label = self._category_to_label()
            self.label_to_cat = self._label_to_category()
            self.gen_to_vec = self._gender_to_vector()
            self.color_to_vec = self._color_to_vector()
            self.season_to_vec = self._season_to_vector()
            self.usage_to_vec = self._usage_to_vector()
        elif mode == 'val':
            self.data = pd.read_csv(os.path.join(root_dir, 'val.csv'))
            self.cat_to_label = cat_to_label
            self.label_to_cat = label_to_cat
            self.gen_to_vec = gen_to_vec
            self.color_to_vec = color_to_vec
            self.season_to_vec = season_to_vec
            self.usage_to_vec = usage_to_vec
        else:
            self.data = pd.read_csv(os.path.join(root_dir, 'test.csv'))
            self.cat_to_label = cat_to_label
            self.label_to_cat = label_to_cat
            self.gen_to_vec = gen_to_vec
            self.color_to_vec = color_to_vec
            self.season_to_vec = season_to_vec
            self.usage_to_vec = usage_to_vec
        self.mode = mode
        with open(cleaned_descrips, 'rb') as f:
            self.cleaned_descrips = pickle.load(f)
        self.lang = lang
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        data = self.data.iloc[idx]
        image_id = str(data['id']) + '.jpg'
        image_path = os.path.join(self.root_dir, 'suffled-images', 'shuffled-images', image_id)
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)

        sample = {
            'image': image,
            'id': data['id'],
            'gender': data['gender'],
            'baseColour': data['baseColour'],
            'season': data['season'],
            'usage': data['usage'],
            'noisyTextDescription': data['noisyTextDescription'],
            'cleaned_descrip': self.lang.encode(self.cleaned_descrips[self.mode][idx]),
            'others': np.concatenate((
                self.gen_to_vec[data['gender']],
                self.color_to_vec[data['baseColour']],
                self.season_to_vec[data['season']],
                self.usage_to_vec[data['usage']]
                ), axis=0)
        }

        if self.mode == 'train' or self.mode == 'val':
            sample['label'] = self.cat_to_label[data['category']]

        return sample
    
    def _category_to_label(self):
        categories = self.data['category'].unique()
        cat_to_label = {}
        for i in range(len(categories)):
            cat_to_label[categories[i]] = i
        return cat_to_label
    
    def _label_to_category(self):
        categories = self.data['category'].unique()
        label_to_cat = []
        for i in range(len(categories)):
            label_to_cat.append(categories[i])
        return np.array(label_to_cat)
    
    def _gender_to_vector(self):
        genders = self.data['gender'].unique()
        gen_to_vec = {}
        for i in range(len(genders)):
            vec = np.zeros(len(genders))
            vec[i] = 1
            gen_to_vec[genders[i]] = vec
        return gen_to_vec
    
    def _color_to_vector(self):
        colors = self.data['baseColour'].unique()
        color_to_vec = {}
        for i in range(len(colors)):
            vec = np.zeros(len(colors))
            vec[i] = 1
            color_to_vec[colors[i]] = vec
        return color_to_vec

    def _season_to_vector(self):
        seasons = self.data['season'].unique()
        season_to_vec = {}
        for i in range(len(seasons)):
            vec = np.zeros(len(seasons))
            vec[i] = 1
            season_to_vec[seasons[i]] = vec
        return season_to_vec
    
    def _usage_to_vector(self):
        usages = self.data['usage'].unique()
        usage_to_vec = {}
        for i in range(len(usages)):
            vec = np.zeros(len(usages))
            vec[i] = 1
            usage_to_vec[usages[i]] = vec
        return usage_to_vec

    def mean_and_std(self):
        mean, std = 0, 0
        N = self.__len__()
        for i in range(N):
            sample = self.__getitem__(i)
            image = np.array(sample['image'])
            mean += image.mean(axis=(0,1))
            std += image.std(axis=(0,1))
        return mean/N, std/N

#################################################################################
########################### Data Augmentation
#################################################################################
def data_aug(input_size):
    data_transforms = {
        'train': transforms.Compose([
            # transforms.RandomResizedCrop(input_size),
            # transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            transforms.Normalize(np.array([208.97083528, 205.12498353, 203.3554041])/255, np.array([51.29295506, 54.37506302, 55.18167259])/255)
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            transforms.Normalize(np.array([208.97083528, 205.12498353, 203.3554041])/255, np.array([51.29295506, 54.37506302, 55.18167259])/255)
        ]),
    }
    return data_transforms

#################################################################################
########################### Balance dataset - not used in this project
#################################################################################
def get_weighted_sampler(dataset):
    label_counts = np.zeros((27))
    label_list = []
    for i in range(len(dataset)):
        label = dataset[i]['label']
        label_list.append(label)
        label_counts[label] += 1
    label_list = np.array(label_list)
    class_weight = 1.0 / (label_counts + 1e-5)
    
    label_weight = class_weight[label_list]

    weighted_sampler = WeightedRandomSampler(
        weights=label_weight,
        num_samples=len(label_weight),
        replacement=True
    )
    return weighted_sampler

#################################################################################
########################### word2vec dataset - one hot vector representation
#################################################################################
class Lang:
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = {}
        self.n_words = 1
        self.max_length = 14

    def addDataset(self, dataset):
        for sentence in dataset:
            self.addSentence(sentence)

    def addSentence(self, sentence):
        for word in sentence:
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def encode(self, sentence, device=torch.device('cuda')):
        tensor = torch.zeros(self.max_length, self.n_words)
        for i in range(self.max_length):
            if i < len(sentence):
                word = sentence[i]
                tensor[i][self.word2index[word]] = 1
            else:
                tensor[i][0] = 1
        return tensor