
import os
import pickle
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# Define data transforms: normalizing and data augmentation
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),  # CIFAR-100 mean and std
])


def unpickle(file):
    """Load the binary file and return it as a Python dictionary"""
    with open(file, 'rb') as f:
        data = pickle.load(f, encoding='bytes')
    return data


def generate_train_testset_cifar10(data_dir):
    # Initialize the training dataset
    train_dataset = CIFAR10CustomDataset(data_dir=data_dir, train=True, transform=transform)
    # Load testing dataset
    test_dataset = CIFAR10CustomDataset(data_dir=data_dir, train=False, transform=transform)
    return train_dataset, test_dataset


def load_data(cifar10, cifar100):
    if cifar100 == True:
        trainset = torchvision.datasets.CIFAR100(root='./data/', train=False, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR100(root='./data/', train=False, download=False, transform=transform)
        # Create data loaders
        trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=1)
        testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=1)
    elif cifar10 == True:

        # Path to the extracted CIFAR-10 data folder
        data_dir = "./cifar10/cifar10"  # Replace this with your actual data directory
        train_set, test_set = generate_train_testset_cifar10(data_dir)

        # Create data loaders
        trainloader = DataLoader(train_set, batch_size=256, shuffle=True, num_workers=1)
        testloader = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=1)
    return trainloader, testloader

class CIFAR10CustomDataset(Dataset):
    def __init__(self, data_dir, train=True, transform=None):
        """
        Args:
            data_dir (string): Directory where the CIFAR-10 data is stored.
            train (bool): If True, load training data; else, load test data.
            transform (callable, optional): A function/transform to apply to the images.
        """
        self.data_dir = data_dir
        self.transform = transform

        if train:
            self.data_batches = [unpickle(os.path.join(data_dir, f"data_batch_{i}")) for i in range(1, 6)]
            self.is_train = True
        else:
            self.data_batches = [unpickle(os.path.join(data_dir, "test_batch"))]
            self.is_train = False
        
        # Extract image and label data
        self.images = []
        self.labels = []
        for batch in self.data_batches:
            self.images.append(batch[b"data"])
            self.labels.append(batch[b"labels"])

        self.images = np.concatenate(self.images, axis=0)
        self.labels = np.concatenate(self.labels, axis=0)

        # Reshape images from 1D array to 3D (32x32 RGB images)
        self.images = self.images.reshape((self.images.shape[0], 3, 32, 32)).transpose(0, 2, 3, 1)

    def __len__(self):
        """Return the number of images in the dataset"""
        return len(self.images)

    def __getitem__(self, idx):
        """Fetch an image and label pair"""
        image = Image.fromarray(self.images[idx])  # Convert numpy array to PIL image
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

