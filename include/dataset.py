import matplotlib.patches as patches
import matplotlib.pyplot as plt
import json
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from utility import *

# alias
from os.path import join as join_paths

''' 
UTILITY FUNCTIONS
'''  
def read_file(file_path):
    # Returns List[dict]: A list of dictionaries representing the data.
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def write_file(output_path, data):
    # Writes a list of dictionaries to a JSON file.
    output_path = output_path 
    with open(output_path, 'w') as file:
        json.dump(data, file, indent=2)  # indent for pretty formatting (optional)
    print(f'Written data to file: {output_path}')
    return 

def match_in_path(path, expression):
    numbers = re.findall(r'\d+', expression)
    if numbers:
        one_number = ''.join(numbers)
    result_string = re.sub(r'\d+', '', expression)
    result_string = result_string.strip()
    match = re.search(fr'\b{expression}\b', path)
    if match and int(one_number)!=None:
        return int(one_number)
    else:
        print(f"Error: No match found for expression '{expression}' in the path.")

'''
WRAPPER FOR DATASET CLASS 
'''
class DataLoaderVisualizer:
    def __init__(self, root, file_path, percentage,  split='train'):
        self.root = root + 'data/images_aligned'
        self.file_path = file_path
        self.split = split
        self.percentage = percentage
        self.data_complete = []
        self.path_structure = self.create_paths()
        self.drivers = self.divide_drivers()
        self.data = self.check_existence()

    def divide_drivers(self):
        list_of_driver_paths = os.listdir(self.root)
        length = len(list_of_driver_paths)   
        if   self.split == 'train':
            drivers = list_of_driver_paths[:length-4]
        elif self.split == 'test':
            drivers = list_of_driver_paths[length-2:]
        elif self.split == 'val':
            drivers = list_of_driver_paths[length-4:length-2]
        elif self.split == 'all':
            drivers = list_of_driver_paths
        else:
            raise ValueError("Invalid split. Use 'train', 'test','val' or 'all'.")
        return drivers

    def check_existence(self):
        if os.path.exists(self.file_path) and str(self.percentage) in self.file_path:
            print("The dataset has already been prepared, ready to use")
            data = read_file(self.file_path)
        else:
            data = self.load_data(self.percentage)
            # write the data to the file if it does not exist
            write_file(self.file_path,data)  
        return data

    def create_paths(self):
        print('Building path structure\n')
        path_structure = {}
        for driver in os.listdir(self.root):
            #print(f'--> Processing for {driver}')
            driver_info = {"road_view": {}, "driver_view": {}}
            # Check the driver folder exists
            driver_path = join_paths(self.root, driver)
            if not os.path.isdir(driver_path) or not os.listdir(driver_path):
                print(f'Skipping empty driver folder: {driver}')
                continue
            # Get paths
            driver_view_path = join_paths(self.root, driver, "driver_view")
            road_view_path = join_paths(self.root, driver, "road_view")
            # List folders
            road_sample_folders = [folder for folder in os.listdir(road_view_path) if os.path.isdir(join_paths(road_view_path, folder))]
            driver_sample_folders = [folder for folder in os.listdir(driver_view_path) if os.path.isdir(join_paths(driver_view_path, folder))]
            # Samples in road view
            for road_sample in road_sample_folders:
                #print(f'--> Processing for road_sample: {road_sample}')
                sample_path = join_paths(road_view_path, road_sample)
                driver_info["road_view"][road_sample] = sample_path
            # Samples in driver view
            for driver_sample in driver_sample_folders:
                #print(f'--> Processing for driver_sample: {driver_sample}')
                sample_path = join_paths(driver_view_path, driver_sample)
                driver_info["driver_view"][driver_sample] = sample_path   
            path_structure[driver] = driver_info
        return path_structure
    
    def load_data(self, percentage):
        print('Loading data in file')
        data = []
        for driver in self.drivers:
            driver_view_samples = self.path_structure[driver]['driver_view']
          
            for sample_name, sample_path in driver_view_samples.items():
                video_images = [img for img in os.listdir(sample_path)]

                # Determine the number of images to load based on the specified percentage
                num_images_to_load = int(len(video_images) * (percentage / 100))
                selected_images = random.sample(video_images, num_images_to_load)
            
                for img in  selected_images:
                    data_item = {}
                    data_item_complete = {}
                    # Save the image path
                    data_item['path'] = join_paths(sample_path, img)
                    data_item_complete['driver path'] = join_paths(sample_path, img)
                    # Save road view path
                    data_item_complete['road view path'] = join_paths(sample_path.replace('driver_view', 'road_view'), img)

                    # Get frame number
                    frame_number = int(img.split("_")[1].split(".")[0])
                    gt_path = sample_path.replace('driver_view', 'road_view') + '.npy'
                    gt_array = np.load(gt_path, allow_pickle=True)
                    sample_number = match_in_path(gt_path, sample_name)
                    frame_number_label, gaze_point, bbox = get_frame_and_point(gt_array, frame_number, sample_number)      
                    assert frame_number == frame_number_label, f"Not the same frame number: frame_number={frame_number}, " \
                                                              f"frame_number_label={frame_number_label}, img_path ={img}, " \
                                                              f"array={gt_path}" 
                    # Save Label
                    data_item['label']  = gaze_point
                    data_item_complete['label']  = gaze_point
                    data_item_complete['bbox']  = list(bbox)  
                    # Append data items to the list
                    data.append(data_item)
                    self.data_complete.append(data_item_complete)
        return data
    
    def __len__(self):
        #assert len(self.data) == len(self.data_complete)
        return len(self.data)

    def visualize_dataset(self):
        for driver in self.drivers:
          all_paths = [item['driver path'] for item in self.data_complete]
          driver_x_paths = find_driver_paths(all_paths, driver)
          driver_img_path = random.sample(driver_x_paths,1)
          driver_img_path = driver_img_path[0]
          road_img_path = driver_img_path.replace('driver_view', 'road_view')
          # Load Images
          driver_photo = cv2.imread(driver_img_path)
          driver_photo = cv2.cvtColor(driver_photo, cv2.COLOR_BGR2RGB)
          road_photo = cv2.imread(road_img_path)
          road_photo = cv2.cvtColor(road_photo, cv2.COLOR_BGR2RGB)
          # Get the bounding box
          matching_item = next((item for item in self.data_complete if item['driver path'] == driver_img_path), None)
          boundbox = matching_item['bbox']
          rect = patches.Rectangle((boundbox[0], boundbox[1]), boundbox[2]-boundbox[0], boundbox[3]-boundbox[1],
                                 linewidth=1, edgecolor='r', facecolor='none')
          # Get the gaze point
          gaze = matching_item['label']
          # Display
          fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(11, 5))
          ax1.imshow(driver_photo)
          ax1.set_title(f'{driver} view photo'), ax1.set_xlabel('W'), ax1.set_ylabel('H')
          ax2.imshow(road_photo)
          ax2.scatter(gaze[0], gaze[1], color='red', marker='x', s=100)
          ax2.add_patch(rect)
          ax2.set_title('road view photo'), ax2.set_xlabel('W'), ax2.set_ylabel('H')
          ax2.set_box_aspect(driver_photo.shape[0] / driver_photo.shape[1])
          plt.show()

'''
PYTORCH DATASET CLASS 
'''
class DGAZEDataset(Dataset):
    def __init__(self, split='train', save_file='train_paths.json', transform = None):
        self.split = split
        self.transform = transform

        if split in save_file:
            self.save_file = save_file
        else:
            raise ValueError("You used the wrong path")

        self.data = read_file(self.save_file) 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]['path']
        label = self.data[idx]['label']

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # OpenCV uses BGR, convert to RGB

        # Assuming self.transform is defined somewhere
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32), img_path
