import os
import torch
import numpy as np
import cv2
import torch.nn.functional as F
import torchvision.transforms as transforms
import glob
import argparse
import pathlib

from model import build_model

# Constructs the argument parser
parser = argparse.ArgumentParser()
parser.add_argument(
    "-w", "--weights",
    default="outputs/best_model.pth",
    help="path to latest checkpoint (default: None)"
)
args = parser.parse_args()

# Constants and other configurations
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 224
CLASS_NAMES = class_names = [
    "ASC_H", "ASC_US", "HSIL", "LSIL", "SCC"
]

# Validation transforms
def get_test_transforms(image_size):
    test_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return test_transforms

def annotate_image(output_class, orig_image):
    class_name = CLASS_NAMES[int(output_class)]
    cv2.putText(
        orig_image,
        f"{class_name}",
        (5, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2,
        lineType=cv2.LINE_AA
    )
    return orig_image

def inference(model, testloader, device, orig_image):
    model.eval()
    counter = 0
    with torch.no_grad():
        counter += 1
        image = testloader
        image = image.to(device)
        outputs = model(image)
    predictions = F.softmax(outputs, dim=1).cpu().numpy()
    output_class = np.argmax(predictions)
    result = annotate_image(output_class, orig_image)
    return result


if __name__ == "__main__":
    weights_path = pathlib.Path(args.weights)
    infer_result_path = os.path.join(
        'outputs', 'inference_results'
    )
    os.makedirs(infer_result_path, exist_ok=True)

    checkpoint = torch.load(weights_path)
    # Load the model.
    model = build_model(
        num_classes=len(CLASS_NAMES)
    ).to(DEVICE)
    model.load_state_dict(checkpoint['state_dict'])

    all_image_paths = glob.glob(os.path.join('data', 'Test', 'SCC', '*'))

    transform = get_test_transforms(IMAGE_SIZE)

    for i, image_path in enumerate(all_image_paths):
        print(f"Inference on image: {i + 1}")
        image = cv2.imread(image_path)
        orig_image = image.copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = transform(image)
        image = torch.unsqueeze(image, 0)
        result = inference(
            model,
            image,
            DEVICE,
            orig_image
        )
        # Save the image to disk.
        image_name = image_path.split(os.path.sep)[-1]
        # cv2.imshow('Image', result)
        # cv2.waitKey(1)
        cv2.imwrite(
            os.path.join(infer_result_path, image_name), result
        )
