from luna_model import resnet18
from luna_data_import import test_loader
import torchvision.transforms as transforms
import torch
import numpy as np
import matplotlib.pyplot as plt
import math

PATH = 'C:/Users/Ed/Desktop/luna/trained_weights.pth'

device = ('cuda' if torch.cuda.is_available() else 'cpu')
resnet18.to(device)
resnet18.load_state_dict(torch.load(PATH))
resnet18.eval()

# arrays to log incorrect predications across test batches
incorrect_images = []
incorrect_labels = []
# var to track number of accurate predictions
test_running_correct = 0

with torch.no_grad():
    for i, data in enumerate(test_loader):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        outputs = resnet18(inputs)
        outputs = outputs.squeeze(-1)

        # return the location in the output tensor of the largest values for each image, since location == label, this returns the predications for each image
        _,preds = torch.max(outputs,1)

        # for a given batch, add up the number of correctly labeled predications and add to running count
        test_running_correct += (preds == labels).sum().item()
        # identify the locations in the predictions tensor that were incorrect vs. test data
        incorrects = np.nonzero(preds.reshape((-1,)) != labels)
        
        # add the incorrect predictions to a numpy array
        incorrect_labels.append(preds[incorrects].cpu().numpy())
        
        # undo the normalization required by the model and restore the original pixel values of the image
        normalized_incorrect_inputs = inputs[incorrects]
        inv_normalize = transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
            std=[1/0.229, 1/0.224, 1/0.255]
        )
        unnormalized_incorrect_inputs = inv_normalize(normalized_incorrect_inputs)

        # add the incorreclty labeled images with their original pixel values to a numpy array
        incorrect_images.append(unnormalized_incorrect_inputs.cpu().numpy())

# print accuracy against test set to screen
print(str((test_running_correct/len(test_loader.dataset))*100) + ' accuracy')

# make a plot containing all incorrectly labeled images and their predictions
# calculate number of rows necessary to display the incorrectly labeled images 

total_incorrect = len(incorrect_labels[0][:])
n_row = math.ceil(total_incorrect/4)
n_col = 4

fig = plt.figure(figsize=(10, 10))

for i in range(total_incorrect):
    fig.add_subplot(n_row, n_col, i + 1)
    plt.imshow(incorrect_images[0][i][0][:][:][:].transpose(1,2,0))
    if incorrect_labels[0][i][0] == 0:
        plt.title('Prediction: Enzo')
    elif  incorrect_labels[0][i][0] == 1:
        plt.title('Prediction: Luna')
    else:  
        plt.title('Prediction: Marvin')
    
plt.show()
