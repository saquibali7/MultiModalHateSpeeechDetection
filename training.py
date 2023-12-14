from imports import *
import EnsembleModel, InceptionNetModel from model
from dataLoader import *



batch_size = 16
learning_rate = 0.001
num_epochs = 10
num_features = 512
num_classes=2

inception_model = InceptionNetModel(num_features)
xlnet = transformers.XLNetModel.from_pretrained("xlnet-base-cased")
bert = BertModel.from_pretrained('bert-base-uncased', return_dict=True)

ensemble_model = EnsembleModel(inception_model, xlnet, bert, num_classes)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.CrossEntropyLoss(reduction='sum')
optimizer = torch.optim.Adam(ensemble_model.parameters(), lr = 3e-4,weight_decay=3e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)

def train_one_epoch(model,dataloader):
    model = ensemble_model.to(device)
    model.train()

    train_loss = 0
    train_acc = 0

    actuals, predictions = [], []

    loop = tqdm(dataloader, total=len(dataloader),desc='Train')

    for b , image, label, bert_encoding, xlnet_encoding in enumerate(loop):
        image = image.to(device)
        image = image.float()
        encoding0 = bert_encoding.to(device)
        encoding1 = xlnet_encoding.to(device)
        label = label.to(device)

        out = model(image,encoding0,encoding1)
        cur_train_loss = criterion(out, label)
        actuals.extend(label.cpu().numpy().astype(int))
        predictions.extend(F.softmax(out, 1).cpu().detach().numpy())
        cur_train_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_loss += cur_train_loss.item()

    scheduler.step()

    predictions = np.array(predictions)
    predicted_labels = predictions.argmax(1)
    accuracy = (predicted_labels == actuals).mean()

    return train_loss/len(dataloader) , accuracy

def valid_one_epoch(model,dataloader):

    model = model.to(device)

    val_loss = 0
    val_acc = 0
    actuals, predictions = [], []

    model.eval()
    with torch.no_grad():

        loop = tqdm(dataloader, total=len(dataloader),desc='Valid')

        for b , image, label, bert_encoding, xlnet_encoding in enumerate(loop):
            image = image.to(device)
            image = image.float()
            encoding0 = bert_encoding.to(device)
            encoding1 = xlnet_encoding.to(device)
            label = label.to(device)

            out = model(image,encoding0,encoding1)

            actuals.extend(label.cpu().numpy().astype(int))
            predictions.extend(F.softmax(out, 1).cpu().detach().numpy())

            cur_valid_loss = criterion(out, label)
            val_loss += cur_valid_loss.item()

            
    predictions = np.array(predictions)
    predicted_labels = predictions.argmax(1)
    accuracy = (predicted_labels == actuals).mean()

    return val_loss/len(dataloader) ,accuracy


NUM_EPOCHS = 50
history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
best_acc = 0.0


for epoch in range(NUM_EPOCHS):

    train_loss , train_acc = train_one_epoch(model=ensemble_model, dataloader=train_loader)
    val_loss , val_acc = valid_one_epoch(model=ensemble_model, dataloader=test_loader)

    print(f"\n Epoch:{epoch + 1} / {NUM_EPOCHS},train loss:{train_loss:.5f}, train acc: {train_acc:.5f}, valid loss:{val_loss:.5f}, valid acc:{val_acc:.5f}")


    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)

    if val_acc > best_acc:
      torch.save(ensemble_model.state_dict(),'best_xlent.pth')   
