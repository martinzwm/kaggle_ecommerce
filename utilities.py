import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import pickle

# For noisyTextDescription cleaning
import nltk
# nltk.download('punkt')
# nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
from scipy import stats

from dataset import *

#################################################################################
########################### Training and Validation
#################################################################################
def train(model, model_name, train_loader, val_loader, loss_fn, optimizer, num_epochs, print_every=100, device=torch.device('cpu')):
    train_loss_list, train_acc_list, val_acc_list = [], [], []
    for epoch in range(num_epochs):
        print('Starting epoch %d / %d' % (epoch + 1, num_epochs))
        total_loss = 0
        model.train()
        for t, sample in enumerate(train_loader):
            x = sample['image'].to(device)
            other_ft = sample['others'].float().to(device)
            text_ft = sample['cleaned_descrip'].to(device)
            y = sample['label'].view(len(x),).long().to(device)
            x_var = Variable(x)
            y_var = Variable(y)
            other_var = Variable(other_ft)
            text_var = Variable(text_ft)

            scores = model(x_var, other_var, text_var)
            loss = loss_fn(scores, y_var)
            total_loss += loss.data
            if (t + 1) % print_every == 0:
                print('t = %d, loss = %.6f' % (t + 1, loss.item()))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if device == torch.device('cuda'):
            train_loss_list.append(total_loss.cpu())
        else:
            train_loss_list.append(total_loss)

        train_acc = check_accuracy(model, train_loader, device=device)
        train_acc_list.append(train_acc)
        val_acc = check_accuracy(model, val_loader, device=device)
        val_acc_list.append(val_acc)
        print('Training accuracy is (%.2f)' % (100 * train_acc))
        print('Validation accuracy is (%.2f)' % (100 * val_acc))

        # If the model is the best so far, save it to ckpt
        if val_acc >= max(val_acc_list):
            torch.save(model, 'best_model_{}.pt'.format(model_name))

    return train_loss_list, train_acc_list, val_acc_list

def check_accuracy(model, loader, device=torch.device('cpu'), debug=False):
    if debug:
        mistakes = torch.empty(0,2).to(device)
    pred_list = torch.empty(0,).to(device)

    num_correct, num_samples = 0, 0
    model.eval()
    with torch.no_grad():
        for t, sample in enumerate(loader):
            x = sample['image'].to(device)
            y = sample['label'].view(len(x),).long().to(device)
            other_ft = sample['others'].float().to(device)
            text_ft = sample['cleaned_descrip'].to(device)
            x_var = Variable(x)
            other_var = Variable(other_ft)
            text_var = Variable(text_ft)
            
            scores = model(x_var, other_var, text_var)
            _, preds = scores.data.max(1)
            num_correct += (preds==y).sum()
            num_samples += preds.size(0)

            pred_list = torch.cat((pred_list, preds), dim=0)

            if debug: # we want to see which categories are the mistakes being made
                mistake_index = preds!=y
                current_mistake = torch.cat((y[mistake_index].view(-1,1), preds[mistake_index].view(-1,1)), dim=1)
                mistakes = torch.cat((mistakes, current_mistake), dim=0)

    acc = float(num_correct) / num_samples
    if debug:
        return acc, mistakes, pred_list
    return acc  

def test_single(model, loader, device=torch.device('cpu')):
    model.eval()
    total_preds = torch.empty(0,).to(device)
    with torch.no_grad():
        for t, sample in enumerate(loader):
            x = sample['image'].to(device)
            other_ft = sample['others'].float().to(device)
            text_ft = sample['cleaned_descrip'].to(device)
            x_var = Variable(x)
            other_var = Variable(other_ft)
            text_var = Variable(text_ft)
            
            scores = model(x_var, other_var, text_var)
            _, preds = scores.data.max(1)

            total_preds = torch.cat((total_preds, preds), dim=0)
    return total_preds

# For a single model
def debug_val(model_name='resnet101'):
    use_heuristics = False
    with open('cleaned_descrips.pickle', 'rb') as f:
        cleaned_descrips = pickle.load(f)
    
    lang = Lang()
    lang.addDataset(cleaned_descrips['train'])
    lang.addDataset(cleaned_descrips['val'])
    lang.addDataset(cleaned_descrips['test'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == torch.device('cuda'):
        batch_size = 256
    else:
        batch_size = 32
    
    input_size = (80,60)
    data_transforms = data_aug(input_size)

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

    model = torch.load('best_model_{}.pt'.format(model_name)).to(device)
    val_acc, val_mistakes, preds = check_accuracy(model, val_loader, device=device, debug=True)
    val_mistakes = val_set.label_to_cat[val_mistakes.cpu().numpy().astype('int32')] # convert mistake labels to categories
    print('Validation accuracy is (%.2f)' % (100 * val_acc))

    val_mistakes = {
        'val_mistakes': [val_mistakes]
    }
    df = pd.DataFrame.from_dict(val_mistakes)
    df.to_pickle('val_mistakes.pickle')

# For a single model
def test(model_name='resnet101'):
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

    model = torch.load('best_model_{}.pt'.format(model_name)).to(device)
    preds = test_single(model, test_loader, device=device)
    heuristics = heuristics_text()
    for i in range(len(preds)):
        heuristic_label = apply_heuristics(heuristics, cleaned_descrips['test'][i])
        if heuristic_label != -1:
            preds[i] = heuristic_label

    preds_category = test_set.label_to_cat[preds]

    preds_category_df = pd.DataFrame(preds_category)
    preds_category_df.to_csv('preds_category.csv', index=False)

# Take the list of model predictions and quickly try different ensemble's accuracy
def best_ensemble():
    all_models = ['resnet152', 'resnet101', 'wide_resnet101_2', 'densenet161', 'vgg19_bn', 'resnext101_32x8d', 'mobilenet_v2', 'densenet121', 'resnet50', 'vgg16_bn']
    preds_dict = pd.read_pickle('preds_dict.pickle').iloc[0][0]
    preds_gt = preds_dict['gt']
    best_acc, best_models = 0, None
    for i in range(1000):
        # N = np.random.randint(3,10)
        N = 8
        model_list = np.random.choice(all_models, size=N, replace=False)
        for i in range(len(model_list)):
            model_name = model_list[i]
            preds_model = preds_dict[model_name].reshape(-1,1)
            if i == 0:
                preds_all = preds_model
            else:
                preds_all = np.concatenate((preds_all, preds_model), axis=1)
        preds_all = torch.from_numpy(preds_all)
        preds_ensemble = torch.mode(preds_all, dim=1)[0].numpy()

        acc = (preds_ensemble==preds_gt).sum() / len(preds_gt)
        print('-------------------------------')
        print(model_list)
        print(acc)
        print('-------------------------------')
        if acc > best_acc:
            best_acc = acc
            best_models = model_list
    print('-----------Best Model----------')
    print(best_models)
    print(best_acc)
    print('-------------------------------')

#################################################################################
########################### Clean noisy text description
#################################################################################
def clean_text(text):
    tokens = word_tokenize(text)
    tokens = [w.lower() for w in tokens]
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    words = [word for word in stripped if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]
    porter = PorterStemmer()
    stemmed = [porter.stem(word) for word in words]
    return stemmed

def clean_text_dataset(dataset, mode='train'):
    max_length = 0
    descriptions = []
    for i in range(len(dataset)):
        text = dataset[i]['noisyTextDescription']
        words = clean_text(text)
        descriptions.append(words)
        length = len(words)
        if length > max_length:
            max_length = length
    print('The max length of the processed description of {} set is {}'.format(mode, max_length))
    return descriptions

def clean_text_main():
    train_set = ImageDataset(mode='train')
    train_descrips = clean_text_dataset(train_set, mode='train')

    val_set = ImageDataset(
        mode='val', cat_to_label=train_set.cat_to_label, 
        label_to_cat=train_set.label_to_cat, gen_to_vec=train_set.gen_to_vec, color_to_vec=train_set.color_to_vec,
        season_to_vec=train_set.season_to_vec, usage_to_vec=train_set.usage_to_vec)
    val_descrips = clean_text_dataset(val_set, mode='val')

    test_set = ImageDataset(
        mode='test', cat_to_label=train_set.cat_to_label, 
        label_to_cat=train_set.label_to_cat, gen_to_vec=train_set.gen_to_vec, color_to_vec=train_set.color_to_vec,
        season_to_vec=train_set.season_to_vec, usage_to_vec=train_set.usage_to_vec)
    test_descrips = clean_text_dataset(test_set, mode='test')

    # save cleaned descrips in pickle file
    cleaned_descrips = {
        'train': train_descrips,
        'val': val_descrips,
        'test': test_descrips
    }
    with open('cleaned_descrips.pickle', 'wb') as f:
        pickle.dump(cleaned_descrips, f)

#################################################################################
########################### Visualization of class and category distribution
#################################################################################
def load_data():
    train_df = pd.read_csv('uw_cs480_fall20/train.csv')
    val_df = pd.read_csv('uw_cs480_fall20/val.csv')
    test_df = pd.read_csv('uw_cs480_fall20/test.csv')
    return train_df, val_df, test_df

def visualize_distribution(train_df, val_df, test_df):
    print('--------Train--------')
    for column in train_df.columns.values:
        if column != 'id' and column != 'noisyTextDescription':
            train_df[column].value_counts().plot(kind='bar')
            plt.show()

    print('--------Val--------')
    for column in val_df.columns.values:
        if column != 'id' and column != 'noisyTextDescription':
            val_df[column].value_counts().plot(kind='bar')
            plt.show()

    print('--------Test--------')
    for column in test_df.columns.values:
        if column != 'id' and column != 'noisyTextDescription':
            test_df[column].value_counts().plot(kind='bar')
            plt.show()

#################################################################################
########################### Heuristics - not used
#################################################################################  
def apply_heuristics(heuristics, text):
    for word in text:
        if word in heuristics:
            return heuristics[word]
    return -1

def heuristics_text():
    categories = [
        'scarve', 'flip flop', 'topwear', 'sandal_notgood', 'bag_notgood', 'sock', 'shoe', 'watch', 'dress_notgood', 'headwear', 'jewellery', 'bottomwear', 'innerwear_notgood', 'wallet', 'belt_notgood', 'saree', 'nail', 'loungewear and nightwear', 'lip_terrible', 'eyewear_ok', 'makeup_terrible', 'tie_notgood', 'fragrance', 'cufflink_notgood', 'free gift', 'apparel set', 'accessorie'
        ]
    heuristics = {}
    for i, category in enumerate(categories):
        heuristics[category] = i
    
    heuristics['shirt'] = 2 # 99.6
    # heuristics['sweater'] = 2
    heuristics['backpack'] = 4 # 98.6
    heuristics['flat'] = 6 # 97.1
    # heuristics['heel'] = 6
    # heuristics['jean'] = 11
    heuristics['bra'] = 12 # 97.0
    heuristics['sunglass'] = 19 # 96.8

    return heuristics

def test_heuristics():
    with open('cleaned_descrips.pickle', 'rb') as f:
        cleaned_descrips = pickle.load(f)
    
    lang = Lang()
    lang.addDataset(cleaned_descrips['train'])
    lang.addDataset(cleaned_descrips['val'])
    lang.addDataset(cleaned_descrips['test'])

    train_set = ImageDataset(mode='train', lang=lang)
    val_set = ImageDataset(
        mode='val', cat_to_label=train_set.cat_to_label, 
        label_to_cat=train_set.label_to_cat, gen_to_vec=train_set.gen_to_vec, color_to_vec=train_set.color_to_vec,
        season_to_vec=train_set.season_to_vec, usage_to_vec=train_set.usage_to_vec,
        lang=lang)
    
    heuristics = heuristics_text()
    metric = {}
    for key in heuristics:
        metric[key] = [0, 0]

    num_correct, num_total = 0, 0
    
    for i in range(len(val_set)):
        for word in cleaned_descrips['val'][i]:
            if word in heuristics:
                if heuristics[word] == val_set[i]['label']:
                    num_correct += 1
                    metric[word][0] += 1
                else:
                    print(i)
                    print(cleaned_descrips['val'][i])
                    print(heuristics[word], val_set[i]['label'])
                num_total += 1
                metric[word][1] += 1
    
    for i in range(len(train_set)):
        for word in cleaned_descrips['train'][i]:
            if word in heuristics:
                if heuristics[word] == train_set[i]['label']:
                    num_correct += 1
                    metric[word][0] += 1
                else:
                    print(i)
                    print(cleaned_descrips['train'][i])
                    print(heuristics[word], train_set[i]['label'])
                num_total += 1
                metric[word][1] += 1

    print('total metric')
    print(num_total)
    print(100.0 * num_correct / num_total)

    print('individual metric')
    for key in metric:
        if metric[key][1] == 0:
            print('{} never appeared'.format(key))
        else:
            print('For {}, total num: {}, correct ratio {}'.format(key, metric[key][1], metric[key][0] / metric[key][1]))
    print(train_set.cat_to_label)

def ensemble_heuristics(preds_ensemble, counts_ensemble, N, texts):
    # heuristics = heuristics_text()
    heuristics = {}
    heuristics['shoe'] = 6
    heuristics['shirt'] = 2
    heuristics['backpack'] = 4
    for i in range(len(preds_ensemble)):
        # count_ensemble = counts_ensemble[i]
        # if count_ensemble > N/2: # if most models agree, this beats heuristics
        #     continue
        text = texts[i]
        heuristics_label = apply_heuristics(heuristics, text)
        if heuristics_label != -1: # override model prediction with heuristics
             preds_ensemble[i] = heuristics_label
    return preds_ensemble