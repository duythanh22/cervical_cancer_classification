import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Required constants
TRAIN_DIR = './data/train'
VALID_DIR = './data/valid'
IMG_SIZE = 224
NUM_WORKER = 4

# Training transforms
def get_train_transforms(image_size):
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(35),
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])
    return train_transform

# Validation transforms
def get_valid_transforms(image_size):
    valid_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return valid_transform

def get_dataset():
    dataset_train = datasets.ImageFolder(
        TRAIN_DIR,
        transform=get_train_transforms(IMG_SIZE)
    )
    dataset_valid = datasets.ImageFolder(
        VALID_DIR,
        transform=get_valid_transforms(IMG_SIZE)
    )
    return dataset_train, dataset_valid, dataset_train.classes

def get_data_loader(dataset_train, dataset_valid, batch_size):
    train_loader = DataLoader(
        dataset_train, batch_size=batch_size,
        shuffle=True, num_workers=NUM_WORKER
    )
    valid_loader = DataLoader(
        dataset_valid, batch_size=batch_size,
        shuffle=False, num_workers=NUM_WORKER
    )
    return train_loader, valid_loader

