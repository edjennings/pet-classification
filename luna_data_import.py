import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import pathlib

train_dataset_location = pathlib.Path('C:/Users/Ed/Desktop/luna/train')  
test_dataset_location = pathlib.Path('C:/Users/Ed/Desktop/luna/test')

# train dataset
# apply resizing and normalization that original model used
train_transform = transforms.Compose([
    transforms.Resize(size=(256,256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])         
])

train_dataset = datasets.ImageFolder(
    root=train_dataset_location,
    transform=train_transform)

train_loader = DataLoader(
    train_dataset, 
    batch_size=30,
    shuffle=True)

# test data set
# apply resizing and normalization that original model used
test_transform = transforms.Compose([
    transforms.Resize(size=(256,256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])         
])

test_dataset = datasets.ImageFolder(
    root=test_dataset_location,
    transform=test_transform
)

test_loader = DataLoader(
    test_dataset, 
    batch_size=30,
    shuffle=False)
