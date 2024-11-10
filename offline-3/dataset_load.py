import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader, random_split
import pickle

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize grayscale channel
])

# Download and load the FashionMNIST dataset
train_data = FashionMNIST(root='data', train=True, transform=transform, download=True)
test_data = FashionMNIST(root='data', train=False, transform=transform, download=True)

# Split training data into train and validation sets
train_size = int(0.8 * len(train_data))
val_size = len(train_data) - train_size
train_dataset, val_dataset = random_split(train_data, [train_size, val_size])

with open('b1.pkl', 'rb') as b1:
  new_test_dataset = pickle.load(b1)

# for sample in new_test_dataset:
#     print('Image size:', sample[0].shape)
#     print('Image label:', sample[1])


# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)
new_test_loader = DataLoader(new_test_dataset, batch_size=64, shuffle=False)


# for dataloader, the unique label count
print('Unique labels in training data:', len(set(train_data.targets.numpy())))
print('Unique labels in test data:', len(set(test_data.targets.numpy())))


unique_labels = set()
unique_labels_count = 0

for images, labels in new_test_loader:
    # print('Image batch dimensions:', images.shape)
    # print('Image label dimensions:', labels.shape)
    # print('Image label values:', labels)
    unique_labels.update(set(labels.numpy()))

    
print('Unique labels in new test data:', len(unique_labels))