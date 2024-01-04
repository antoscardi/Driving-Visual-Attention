import matplotlib.patches as patches
import matplotlib.pyplot as plt
import json
import base64
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from utility import *
from extract_features import*

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
    def __init__(self, root, file_path, percentage, predictor, face_detector, headpose_extractor,split='train'):
        self.root = root + 'data/images_aligned'
        self.file_path = file_path
        self.split = split
        self.percentage = percentage
        self.face_detector = face_detector
        self.predictor = predictor
        self.headpose_extractor = headpose_extractor
        self.path_structure = self.create_paths()
        self.drivers = self.divide_drivers()
        self.data, self.data_complete = self.check_existence()

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
            data_complete = read_file(self.file_path + 'complete')
        else:
            data, data_complete = self.load_data(self.percentage)
            # write the data to the file if it does not exist
            write_file(self.file_path, data)
            write_file(self.file_path + 'complete', data_complete)   
        return data, data_complete

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
        print('Loading data')
        data = []
        data_complete = []
        for driver in tqdm(self.drivers):
            driver_view_samples = self.path_structure[driver]['driver_view']  
            for sample_name, sample_path in driver_view_samples.items():
                video_images = [img for img in os.listdir(sample_path)]
                # Determine the number of images to load based on the specified percentage
                num_images_to_load = int(len(video_images) * (percentage / 100))
                selected_images = random.sample(video_images, num_images_to_load)
                for img in  selected_images:
                    data_item = {}
                    data_item_complete = {}
                    # Get frame number
                    frame_number = int(img.split("_")[1].split(".")[0])
                    gt_path = sample_path.replace('driver_view', 'road_view') + '.npy'
                    gt_array = np.load(gt_path, allow_pickle=True)
                    sample_number = match_in_path(gt_path, sample_name)
                    frame_number_label, gaze_point, bbox = get_frame_and_point(gt_array, frame_number, sample_number)      
                    assert frame_number == frame_number_label, f"Not the same frame number: frame_number={frame_number}, " \
                                                              f"frame_number_label={frame_number_label}, img_path ={img}, " \
                                                              f"array={gt_path}" 
                    # Load image
                    image = cv2.imread(join_paths(sample_path, img))
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # OpenCV uses BGR, convert to RGB
                    # Get eyes
                    result = get_eyes(image, self.predictor, self.face_detector)
                    if result is None:
                        # Skip the current image
                        print(f"Skipping this image:{join_paths(sample_path, img)}")
                        break
                    else:
                        eye_left, pupil_left, pupil_right = result[0], result[1], result[2]
                        # Save the image path
                        data_item['path'] = join_paths(sample_path, img)
                        data_item_complete['driver path'] = join_paths(sample_path, img)
                        # Save road view path
                        data_item_complete['road view path'] = join_paths(sample_path.replace('driver_view','road_view'), img)
                        # Get headpose angles
                        pitch,yaw,roll = get_headpose(image, self.headpose_extractor)                    
                        data_item['feature list'] = [float(pitch), float(yaw), float(roll), float(pupil_left[0]), float(pupil_left[1]), float(pupil_right[0]), float(pupil_right[1])]
                        # Save Label
                        data_item['label']  = gaze_point
                        data_item_complete['label']  = gaze_point
                        data_item_complete['bbox']  = list(bbox) 
                        # Encode the image of the left-eye
                        # Convert the image to a string using base64 encoding
                        eye_left_str = base64.b64encode(cv2.imencode('.jpg', eye_left)[1]).decode('utf-8')
                        data_item['eye left'] = eye_left_str
                        # Append data items to the list
                        data.append(data_item)
                        data_complete.append(data_item_complete)
        return data, data_complete
    
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
          # Calculate landmarks and extract face and eye patches
          face, eye_left, eye_right, pupil_left, pupil_right,  landmarks = get_face_n_eyes(driver_photo, self.face_detector, self.predictor)
          for (x, y) in landmarks:
              cv2.circle(driver_photo, (x, y), 10, (0, 255, 0), -1)
          # Get headpose and plot it on the face 
          face = get_headpose(face, self.headpose_extractor, doPlot = True)
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
          cv2.circle(driver_photo, pupil_left, 10, (255, 0, 0), -1)
          fig, (ax1, ax2, ax3, ax4) = plt.subplots(ncols=4, figsize=(20, 10))
          ax1.imshow(driver_photo)
          ax1.set_title(f'{driver} view photo'), ax1.set_xlabel('W'), ax1.set_ylabel('H')
          ax2.imshow(road_photo)
          ax2.scatter(gaze[0], gaze[1], color='red', marker='x', s=100)
          ax2.add_patch(rect)
          ax2.set_title('road view photo'), ax2.set_xlabel('W'), ax2.set_ylabel('H')
          ax2.set_box_aspect(driver_photo.shape[0] / driver_photo.shape[1])
          ax3.imshow(eye_left)
          ax3.set_title('left eye'), ax3.set_xlabel('W'), ax3.set_ylabel('H')
          ax3.set_box_aspect(driver_photo.shape[0] / driver_photo.shape[1])
          ax4.imshow(face)
          ax4.set_title('cropped face'), ax4.set_xlabel('W'), ax4.set_ylabel('H')
          ax4.set_box_aspect(driver_photo.shape[0] / driver_photo.shape[1])
          plt.show()

'''
PYTORCH DATASET CLASS 
'''
class DGAZEDataset(Dataset):
    def __init__(self, split='train',save_file='train_paths.json', transform = None):
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
            label = self.data[idx]['label']
            img_path = self.data[idx]['path']
            eye_left_encoded = self.data[idx]['eye left']
            additional_features = self.data[idx]['feature list']
            # Decode the image
            eye_left_decoded = base64.b64decode(eye_left_encoded)
            eye_left_array = np.frombuffer(eye_left_decoded, dtype=np.uint8)
            eye_left = cv2.imdecode(eye_left_array, flags=cv2.IMREAD_COLOR)

            if self.transform:
                eye_left= self.transform(eye_left)

            return  eye_left, torch.tensor(additional_features, dtype=torch.float32) , torch.tensor(label, dtype=torch.float32), img_path  
