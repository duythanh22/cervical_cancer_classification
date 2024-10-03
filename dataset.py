import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Required constants
TRAIN_DIR = './dataset_final/train'
VALID_DIR = './dataset_final/dev'
IMG_SIZE = 224
NUM_WORKER = 4

# Training transforms
def get_train_transform(image_size):
    train_transform = transforms.Compose([
        transforms.RandomRotation(degrees=(-30, 30)),  # Rotation with smaller angle to preserve structure
        transforms.RandomHorizontalFlip(p=0.5),  # Increased probability to add more variance
        transforms.RandomVerticalFlip(p=0.5),  # Increased probability for vertical flip
        transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.5),  # Slight sharpening adjustment
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Add color variation
        transforms.Resize((image_size, image_size)),  # Resize to consistent size
        transforms.RandomResizedCrop(size=(image_size, image_size), scale=(0.8, 1.0)),  # Adjusted scale for large cells
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return train_transform

# Validation transforms
def get_valid_transform(image_size):
    valid_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return valid_transformm

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

