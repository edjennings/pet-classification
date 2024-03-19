import torch.optim as optim
from luna_model import resnet18
from luna_data_import import train_loader
import torch
import matplotlib.pyplot as plt

# set learning rate, optimizer, and loss function
lr = 1e-3
optimizer = optim.Adam(resnet18.parameters(),lr=lr)
loss_func = torch.nn.CrossEntropyLoss()

# load model to cuda if available
device = ('cuda' if torch.cuda.is_available() else 'cpu')
resnet18.to(device)

# number of training epochs
epochs = 35

# arrays to keep track of per epoch loss and accuracy on training set
train_loss = []
train_acc = []

for epoch in range(epochs):

    # initialize loss and correct to 0 for current epoch
    epoch_running_loss = 0.0 
    epoch_running_correct = 0

    # iterate through the batches of the training set for the given epoch
    for i, data in enumerate(train_loader):
        
        resnet18.train()
    
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
       
        outputs = resnet18(inputs)
        outputs = outputs.squeeze(-1)

        loss = loss_func(outputs,labels)
        loss.backward()

        optimizer.step()

        _,preds = torch.max(outputs,1)
        
        # add loss from batch to the epoch running loss
        epoch_running_loss += loss.item()
        # add number of correctly labeled predictions from batch to the epoch running correct
        epoch_running_correct += (preds == labels).sum().item()

    # add the avg. loss per batch as the loss metric for a given epoch to the array
    train_loss.append((epoch_running_loss/len(train_loader)))
    # add the accuracy of the epoch to the array
    train_acc.append((epoch_running_correct/len(train_loader.dataset))*100)

# generate and save per epoch loss and accuracy graphs 
print('Finished Training')
plt.figure(1)
plt.plot(train_loss, label = 'Training Loss per Epoch')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.savefig('C:/Users/Ed/Desktop/luna/training_loss.png')

plt.figure(2)
plt.plot(train_acc, label = 'Training Accuracy per Epoch')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.savefig('C:/Users/Ed/Desktop/luna/training_accuracy.png')

print('Graphs saved')

# save model
torch.save(resnet18.state_dict(),'C:/Users/Ed/Desktop/luna/trained_weights.pth')

print('Model Saved')
