import random
import os
import numpy as np
import re
from tqdm import tqdm
import torch
import cv2
from torchvision import transforms
import wandb
from torchvision import utils
import torch.nn as nn
import dlib
import copy


# https://gist.github.com/ihoromi4/b681a9088f348942b01711f251e5f964
def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_bbox_middle_point(bbox):
    x1, y1, x2, y2 = bbox
    middle_x = (x1 + x2) / 2
    middle_y = (y1 + y2) / 2
    return np.array([middle_x, middle_y])

def find_driver_paths(all_paths, driver):
    driver_x_paths = []
    pattern = re.compile(r'driver\d+')
    for path in all_paths:
        match = pattern.search(path)
        if match and match.group() == driver:
            driver_x_paths.append(path)
    return driver_x_paths

def get_frame_and_point(data_array, idx, sample_number):
    count = 0
    for row in data_array:
        if sample_number >= 10:
            # frame_number = int(row[0].replace('frame', ''))
            bbox = row[1:]
            # Return only when you get to the correct frame_id
            if count == idx:
                return count, get_bbox_middle_point(bbox).tolist(), bbox
            count += 1
        else:
            if count == idx:
                return count, row.tolist(), np.zeros(4)  # there is no bbox, use 4 zeros
            count += 1

    # If no matching frame is found
    print(f"Error: No matching frame found for idx={idx} and sample_number={sample_number}.")
    return None

class CropTransform:
    def __init__(self, crop_params):
        self.crop_params = crop_params

    def __call__(self, img):
        return transforms.functional.crop(img, *self.crop_params)
    
def log_random_images(example_images, random_percentage=0.2):
    # Return list of randomly selected images to log
    num_examples = len(example_images)
    num_examples_to_log = int(random_percentage * num_examples)

    random_indices = random.sample(range(num_examples), num_examples_to_log)

    return random_indices

def mark_image(path, pred_x, true_x):
    # Load image
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Overlay "x" markings on the road_view image
    image = cv2.drawMarker(
        image,
        pred_x, 
        color=(255, 0, 0),  # Red color for predicted gaze point
        markerType=cv2.MARKER_CROSS,
        thickness=2,
        markerSize=40,
    )

    image = cv2.drawMarker(
        image,
        true_x,
        color=(0, 0, 255),  # Blue color for true gaze point
        markerType=cv2.MARKER_CROSS,
        thickness=2,
        markerSize=40,
    )

    return image

def log_image(dataloader, model, device, percentage=None):
    with torch.no_grad():
        model.eval()
        # Log example images and predictions 
        example_batch = next(iter(dataloader))
        eyes, features , example_labels, img_paths = example_batch
        # Calculate gaze point prediction
        example_predictions = model(eyes.to(device), features.to(device))
        # Log a single random image with predictions
        random_index = random.randint(0, len(eyes) - 1)
        pred, label, img_path = (
            example_predictions[random_index].cpu(),
            example_labels[random_index].cpu(),
            img_paths[random_index],
        )
        # Load the road_view image
        respective_road_view = img_path.replace('driver_view', 'road_view')
        
        # Mark the road-view with the prediction
        road_view_image = mark_image(respective_road_view, tuple(pred.numpy().astype(int)), tuple(label.numpy().astype(int)))

        # Log the annotated road_view image to WandB
        image = wandb.Image(road_view_image, caption=f"Prediction: {pred}, Actual: {label}")
        wandb.log({"Validation Examples 1 per Epoch": image})


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])