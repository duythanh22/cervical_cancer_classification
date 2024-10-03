import torch
import matplotlib
import matplotlib.pyplot as plt
import os

matplotlib.style.use('ggplot')


class SaveBestModel:
    def __init__(self, best_valid_loss=float('inf')):
        self.best_valid_loss = best_valid_loss

    def __call__(self, current_valid_loss, epoch, model, out_dir, name):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"\nSaving best model for epoch: {epoch + 1}\n")
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
            }, os.path.join(out_dir, 'best_' + name + '.pth'))

class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0, path='checkpoint.pth'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        self.path = path
        self.save_best_model = SaveBestModel()

    def __call__(self, val_loss, epoch, model, optimizer, criterion, out_dir, name):
        score = -val_loss

        # Check if the validation loss has improved
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, epoch, model, optimizer, criterion, out_dir, name)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, epoch, model, optimizer, criterion, out_dir, name)
            self.counter = 0

    def save_checkpoint(self, val_loss, epoch, model, optimizer, criterion, out_dir, name):
        """Save model when validation loss decreases."""
        if self.verbose:
            print(f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...")
        self.save_best_model(val_loss, epoch, model, out_dir, name)  # Save best model here
        self.val_loss_min = val_loss


def save_model(epochs, model, optimizer, criterion, out_dir, name):
    torch.save({
        'epochs': epochs,
        'state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': criterion,
    }, os.path.join(out_dir, name+'.pth'))


def save_plots(train_acc, valid_acc, train_loss, valid_loss, out_dir):
    # Accuracy plot
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_acc, color='tab:blue', linestyle='-',
        label='train accuracy'
    )
    plt.plot(
        valid_acc, color='tab:red', linestyle='-',
        label='valid accuracy'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(out_dir, 'accuracy.png'))

    # Loss plot
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_loss, color='tab:blue', linestyle='-',
        label='train loss'
    )
    plt.plot(
        valid_loss, color='tab:blue', linestyle='-',
        label='valid loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(out_dir, 'loss.png'))

