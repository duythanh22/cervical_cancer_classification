
import torchvision.models as models
import torch.nn as nn
import torch

def build_model(pretrained=True, fine_tune=False, num_classes=5):
    if pretrained:
        print('[INFO]: Loading pre-trained weights')
        weights = models.DenseNet121_Weights.IMAGENET1K_V1
    else:
        print('[INFO]: Not loading pre-trained weights')
        weights = None

    model = models.densenet121(weights=weights)

    # Freeze or fine-tune layers
    if fine_tune is True:
        print('[INFO]: Fine-tuning all layers...')
        for param in model.parameters():
            param.requires_grad = True
    if fine_tune is False:
        print('[INFO]: Freezing hidden layers...')
        for param in model.parameters():
            param.requires_grad = False

    # Replace the classifier
    num_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )

    return model


if __name__ == '__main__':
    model = build_model()
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(total_params)
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f"{total_trainable_params}:, training parameters.")
