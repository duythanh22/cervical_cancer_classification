import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import os

from tqdm.auto import tqdm
from model import build_model
from dataset import get_dataset, get_data_loader
from utils import save_model, save_plots, SaveBestModel, EarlyStopping
from torch.optim.lr_scheduler import MultiStepLR

seed = 22
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

# Construct the argument parser.
parser = argparse.ArgumentParser()
parser.add_argument(
    '-e', '--epochs',
    type=int,
    default=10,
    help='Number of epochs to train our network for'
)
parser.add_argument(
    '-lr', '--learning_rate',
    type=float,
    dest='learning_rate',
    default=0.001,
    help='Learning rate for training the model'
)
parser.add_argument(
    '-b', '--batch_size',
    dest='batch_size',
    default=32,
    type=int
)
parser.add_argument(
    '--save-name',
    dest='save_name',
    default='model',
    help='file name of the final model to save'
)

parser.add_argument(
    '--fine-tune',
    dest='fine_tune',
    default='False',
    help='Fine tuning or extractor'
)

args = vars(parser.parse_args())


# Training function
def train(model, trainloader, optimizer, criterion):
    model.train()
    print('Training started...')
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0
    for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
        counter += 1
        image, labels = data
        image = image.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        # Forward pass
        outputs = model(image)
        # Calculate the loss
        loss = criterion(outputs, labels)
        train_running_loss += loss.item()
        # Calculate the accuracy
        _, preds = torch.max(outputs.data, 1)
        train_running_correct += (preds == labels).sum().item()
        # Backpropagation
        loss.backward()
        # Update the weights
        optimizer.step()

    # Loss and accuracy for the complete epoch
    epoch_loss = train_running_loss / counter
    epoch_acc = 100. * (train_running_correct / len(trainloader.dataset))
    return epoch_loss, epoch_acc


# Validation function
def validate(model, testloader, criterion, class_names):
    model.eval()
    print("Validation started...")
    valid_running_loss = 0.0
    valid_running_correct = 0
    counter = 0

    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            counter += 1
            image, labels = data
            image = image.to(device)
            labels = labels.to(device)
            # Forward pass
            outputs = model(image)
            # Caculate the loss
            loss = criterion(outputs, labels)
            valid_running_loss += loss.item()
            # Calculate accuracy
            _, preds = torch.max(outputs.data, 1)
            valid_running_correct += (preds == labels).sum().item()

        #
        epoch_loss = valid_running_loss / counter
        epoch_acc = 100. * (valid_running_correct / len(testloader.dataset))
        return epoch_loss, epoch_acc


if __name__ == '__main__':
    out_dir = os.path.join('outputs')
    os.makedirs(out_dir, exist_ok=True)
    #
    dataset_train, dataset_valid, dataset_classes = get_dataset()
    print(f"[INFO]: Number of training images: {len(dataset_train)}")
    print(f"[INFO]: Number of validation images: {len(dataset_valid)}")
    print(f"[INFO]: Classes: {len(dataset_classes)}")

    train_loader, valid_loader = get_data_loader(
        dataset_train, dataset_valid, batch_size=args['batch_size']
    )

    lr = args['learning_rate']
    epochs = args['epochs']
    fine_tune = args['fine_tune']
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO]: Using device: {device}")
    print(f"[INFO]: Learning rate: {lr}")
    print(f"[INFO]: Epochs: {epochs}")
    print(f"[INFO]: Fine tune: {fine_tune}\n")

    model = build_model(
        pretrained=True, fine_tune=fine_tune, num_classes=len(dataset_classes)
    ).to(device)
    # print(model)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params}:, total params.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f"{total_trainable_params}:, training params.")

    save_name = args['save_name']
    # Optimizer
    optimizer = optim.Adam(model.parameters())
    # Loss function
    criterion = nn.CrossEntropyLoss()
    # Initialize best
    save_best_model = SaveBestModel()
    # Early stopping
    early_stopping = EarlyStopping(patience=5, verbose=True, path=os.path.join(out_dir, f'{save_name}_checkpoint.pt'))
    # Scheduler
    scheduler = MultiStepLR(
        optimizer, milestones=[5], gamma=0.1, verbose=True
    )

    train_loss, valid_loss = [], []
    train_acc, valid_acc = [], []

    for epoch in range(epochs):
        print(f"[INFO] Epoch {epoch + 1} of {epochs}")
        train_epoch_loss, train_epoch_acc = train(model, train_loader, optimizer, criterion)
        valid_epoch_loss, valid_epoch_acc = validate(model, valid_loader, criterion, dataset_classes)

        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        train_acc.append(train_epoch_acc)
        valid_acc.append(valid_epoch_acc)

        print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
        print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")

        # Early stopping based on validation loss and saving best model
        early_stopping(valid_epoch_loss, epoch, model, optimizer, criterion, out_dir, args['save_name'])

        if early_stopping.early_stop:
            print("Early stopping triggered!!!")
            break

        print('-' * 50)
        scheduler.step()

    save_model(epochs, model, optimizer, criterion, out_dir, args['save_name'])
    save_plots(train_acc, valid_acc, train_loss, valid_loss, out_dir)
    print("Training complete!")


