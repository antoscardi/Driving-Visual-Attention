import random
import os
import numpy as np
import re
import torch

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
    return middle_x, middle_y

def find_driver_paths(all_paths, driver):
    driver_x_paths = []
    pattern = re.compile(r'driver\d+')
    for path in all_paths:
        match = pattern.search(path)
        if match and match.group() == driver:
            driver_x_paths.append(path)
    return driver_x_paths

def get_frame_and_point(data_array, idx, sample_number):
    count = 1
    for row in data_array:
        if sample_number > 10:
            frame_number = int(row[0].replace('frame', ''))
            bbox = row[1:]
            # Return only when you get to the correct frame_id
            if count == idx:
                return count, get_bbox_middle_point(bbox), bbox
            count += 1
        else:
            if count == idx:
                return count, row , np.zeros(4) # there is no bbox use 4 zeros
            count += 1

class CropTransform:
    def __init__(self, crop_params):
        self.crop_params = crop_params

    def __call__(self, img):
        return transforms.functional.crop(img, *self.crop_params)