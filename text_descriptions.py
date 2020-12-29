'''
This is used when testing LSTM on the text descriptions w/o image data.
This is not used in the final model.
Packages and dependencies need to set up before using this file.
'''

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_embeddings):
        super(LSTM, self).__init__()

        self.hidden_size = hidden_size
        self.embedding = nn.Linear(num_embeddings, input_size)
        self.lstm_cell = nn.LSTMCell(input_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o_2 = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        hidden = self.initHidden(input)
        for i in range(input.size(1)):
            scores, hidden = self._forward(input[:,i], hidden)
        return scores

    def _forward(self, input, hidden_and_cell):
        hidden = hidden_and_cell[0]
        input = self.embedding(input)
        combined = torch.cat((input, hidden), 1)
        hidden_and_cell = self.lstm_cell(input, hidden_and_cell)
        output = self.i2o(combined)
        output = self.i2o_2(output)
        return output, hidden_and_cell

    def initHidden(self, input):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batch_size = input.size(0)
        return (torch.zeros(batch_size, self.hidden_size).to(device), torch.zeros(batch_size, self.hidden_size).to(device))

def descrip_processor():
    num_classes = 27
    num_epochs = 10
    input_size=(80,60)

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

    model = LSTM(800, 800, num_classes, lang.n_words).to(device)
    print(model)

    #################################################################################
    ########################### Data Loader
    #################################################################################
    data_transforms = data_aug(input_size)
    train_set = ImageDataset(mode='train', transform=data_transforms['train'], lang=lang)
    val_set = ImageDataset(
        mode='val', transform=data_transforms['val'], cat_to_label=train_set.cat_to_label, 
        label_to_cat=train_set.label_to_cat, gen_to_vec=train_set.gen_to_vec, color_to_vec=train_set.color_to_vec,
        season_to_vec=train_set.season_to_vec, usage_to_vec=train_set.usage_to_vec, lang=lang)
    
    # hidden = model.initHidden()
    # encoded_input = lang.encode(train_set[0]['cleaned_descrip'])
    # print(model(encoded_input[0], hidden))

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
    train_loss_list, train_acc_list, val_acc_list = train_descrips(model, train_loader, val_loader, loss_fn, optimizer, num_epochs, print_every=10, device=device)

    # Save train loss and val loss profile
    train_loss_profile = {
        'train_loss': np.array(train_loss_list),
        'train_acc': np.array(train_acc_list),
        'val_acc': np.array(val_acc_list)
    }
    df = pd.DataFrame.from_dict(train_loss_profile)
    df.to_pickle('train_profile_lstm.pickle')

def check_accuracy_descrips(model, loader, device=torch.device('cpu')):
    num_correct, num_samples = 0, 0
    model.eval()
    with torch.no_grad():
        for t, sample in enumerate(loader):
            x = sample['cleaned_descrip'].to(device)
            y = sample['label'].view(len(x),).long().to(device)
            x_var = Variable(x)
            y_var = Variable(y)
            
            scores = model(x_var)
            _, preds = scores.data.max(1)
            num_correct += (preds==y).sum()
            num_samples += preds.size(0)

        acc = float(num_correct) / num_samples
    return acc     

def train_descrips(model, train_loader, val_loader, loss_fn, optimizer, num_epochs, print_every=100, device=torch.device('cpu')):
    train_loss_list, train_acc_list, val_acc_list = [], [], []
    for epoch in range(num_epochs):
        print('Starting epoch %d / %d' % (epoch + 1, num_epochs))
        total_loss = 0
        model.train()
        for t, sample in enumerate(train_loader):
            x = sample['cleaned_descrip'].to(device)
            y = sample['label'].view(len(x),).long().to(device)
            x_var = Variable(x)
            y_var = Variable(y)
            scores = model(x_var)

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

        train_acc = check_accuracy_descrips(model, train_loader, device=device)
        train_acc_list.append(train_acc)
        val_acc = check_accuracy_descrips(model, val_loader, device=device)
        val_acc_list.append(val_acc)
        print('Training accuracy is (%.2f)' % (100 * train_acc))
        print('Validation accuracy is (%.2f)' % (100 * val_acc))

        # If the model is the best so far, save it to ckpt
        if val_acc >= max(val_acc_list):
            torch.save(model, 'best_model_lstm.pt')

    return train_loss_list, train_acc_list, val_acc_list