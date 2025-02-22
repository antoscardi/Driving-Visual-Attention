import matplotlib.patches as patches
import matplotlib.pyplot as plt
import json
import ijson
import base64
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

def read_big_file(file_path):
    data = []
    with open(file_path, "rb") as f:
        # Create an iterator to parse the JSON file incrementally
        objects = ijson.items(f, "item")     
        for obj in objects:
            # Extract the required information from each dictionary
            label = obj.get('label')
            img_path = obj.get('path')
            eye_left_encoded = obj.get('eye left')
            additional_features = obj.get('feature list')
            bbox = obj.get('bbox')
            face_encoded = obj.get('face') 
            nose_position = obj.get('nose position')  
            corner_eyes = obj.get('corner eyes')
            # Create a dictionary with the extracted information
            entry = {
                'label': label,
                'path': img_path,
                'eye left': eye_left_encoded,
                'feature list': additional_features,
                'bbox': bbox,
                'face': face_encoded,
                'nose position': nose_position,
                'corner eyes': corner_eyes
            }
            # Append the entry to the data list
            data.append(entry)
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
    def __init__(self, root, file_path, percentage, predictor, face_detector, headpose_extractor,split='train',big_file = False):
        self.root = root + 'data/images_aligned'
        self.file_path = file_path
        self.split = split
        self.big_file = big_file
        self.percentage = percentage
        self.face_detector = face_detector
        self.predictor = predictor
        self.headpose_extractor = headpose_extractor
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
            if self.big_file:
                data = read_big_file(self.file_path)
            else:
                data = read_file(self.file_path)
        else:
            data = self.load_data(self.percentage)
            # write the data to the file if it does not exist
            write_file(self.file_path, data) 
        return data

    def create_paths(self):
        print('Building path structure')
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
        for driver in tqdm(self.drivers):
            count = 0
            driver_view_samples = self.path_structure[driver]['driver_view']  
            for sample_name, sample_path in driver_view_samples.items():
                video_images = np.array(os.listdir(sample_path))
                # Determine the number of images to load based on the specified percentage
                num_images_to_load = int(len(video_images) * (percentage / 100))
                selected_images = np.random.choice(video_images, num_images_to_load, replace=False)
                for img in  selected_images:
                    # Get frame number
                    frame_number = int(img.split("_")[1].split(".")[0])
                    gt_path = sample_path.replace('driver_view', 'road_view') + '.npy'
                    gt_array = np.load(gt_path, allow_pickle=True)
                    sample_number = match_in_path(gt_path, sample_name)
                    frame_number_label, gaze_point, bbox = get_frame_and_point(gt_array, frame_number, sample_number)      
                    assert frame_number == frame_number_label, f"Not the same frame number: frame_number={frame_number}, " \
                                                               f"frame_number_label={frame_number_label}, img_path ={img}, " \
                                                               f"array={gt_path}" 
                    # Get eyes
                    result = get_features(join_paths(sample_path, img), self.predictor, self.face_detector)
                    if result is None:
                        count += 1
                        # Skip the current image
                        continue
                    else:
                        eye_left, pupil_left, pupil_right,face_image_cropped, nose_position, corners_of_eyes = result[0], result[1], result[2], result[3],result[4],result[5]
                        # Get headpose angles
                        pitch,yaw,roll = get_headpose(join_paths(sample_path, img), self.headpose_extractor)                    
                        # Append data items to the list
                    data.append({
                        'path': join_paths(sample_path, img),
                        'road view path': join_paths(sample_path.replace('driver_view', 'road_view'), img),
                        'feature list': [float(pitch), float(yaw), float(roll), float(pupil_left[0]), float(pupil_left[1]), float(pupil_right[0]), float(pupil_right[1])],
                        'label': gaze_point,
                        'bbox': list(bbox),
                        'eye left': base64.b64encode(cv2.imencode('.jpg', eye_left)[1]).decode('utf-8'),
                        'face': base64.b64encode(cv2.imencode('.jpg', face_image_cropped)[1]).decode('utf-8'),
                        'nose position': [float(nose_position[0]), float(nose_position[1])],
                        'corner eyes': [float(corners_of_eyes[0][0]), float(corners_of_eyes[0][1]), float(corners_of_eyes[1][0]), float(corners_of_eyes[1][1]), float(corners_of_eyes[2][0]), float(corners_of_eyes[2][1]), float(corners_of_eyes[3][0]), float(corners_of_eyes[3][1])]
                    })
            if count !=0:
                print(f"In {driver} I skipped {count} images")
        return data
    
    def __len__(self):
        #assert len(self.data) == len(self.data_complete)
        return len(self.data)

    def visualize_dataset(self):
        for driver in self.drivers:
          seed_everything(42)
          all_paths = [item['path'] for item in self.data]
          driver_x_paths = find_driver_paths(all_paths, driver)
          driver_img_path = random.sample(driver_x_paths,1)
          driver_img_path = driver_img_path[0]
          road_img_path = driver_img_path.replace('driver_view', 'road_view')
          # Load Images
          driver_photo = cv2.imread(driver_img_path)
          driver_photo = cv2.cvtColor(driver_photo, cv2.COLOR_BGR2RGB)
          # Calculate landmarks and extract face and eye patches
          result = get_features(driver_img_path, self.predictor,self.face_detector, doPlot=True)
          if result:
              img, photo_draw, left_eye = result[0], result[1], result[2]
          else:
              continue
          # Get headpose and plot it on the face 
          #face = get_headpose(driver_img_path, self.headpose_extractor, doPlot = True)
          road_photo = cv2.imread(road_img_path)
          road_photo = cv2.cvtColor(road_photo, cv2.COLOR_BGR2RGB)
          # Get the bounding box
          matching_item = next((item for item in self.data if item['path'] == driver_img_path), None)
          boundbox = matching_item['bbox']
          rect = patches.Rectangle((boundbox[0], boundbox[1]), boundbox[2]-boundbox[0], boundbox[3]-boundbox[1],
                                 linewidth=1, edgecolor='r', facecolor='none')
          # Get the gaze point
          gaze = matching_item['label']
          # Display
          fig, (ax1, ax2, ax3, ax4) = plt.subplots(ncols=4, figsize=(20, 10))
          ax1.imshow(photo_draw)
          ax1.set_title(f'{driver} view photo'), ax1.set_xlabel('W'), ax1.set_ylabel('H')
          ax2.imshow(road_photo)
          ax2.scatter(gaze[0], gaze[1], color='red', marker='x', s=100)
          ax2.add_patch(rect)
          ax2.set_title('road view photo'), ax2.set_xlabel('W'), ax2.set_ylabel('H')
          ax2.set_box_aspect(driver_photo.shape[0] / driver_photo.shape[1])
          ax3.imshow(img)
          ax3.set_title('additional image in dataset'), ax3.set_xlabel('W'), ax3.set_ylabel('H')
          ax3.set_box_aspect(driver_photo.shape[0] / driver_photo.shape[1])
          ax4.imshow(left_eye)
          ax4.set_title('left eye'), ax4.set_xlabel('W'), ax4.set_ylabel('H')
          ax4.set_box_aspect(driver_photo.shape[0] / driver_photo.shape[1])
          plt.show()
          seed_everything(42)

'''
PYTORCH DATASET CLASS 
'''
class DGAZEDataset(Dataset):
    def __init__(self, split='train', save_file='train_paths.json', transform = None, big_file=False):
        self.split = split
        self.transform = transform
        self.big_file = big_file

        if split in save_file:
            self.save_file = save_file
        else:
            raise ValueError("You used the wrong path")
        if self.big_file:
            self.data = read_big_file(self.save_file)
        else:
            self.data = read_file(self.save_file) 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
            label = self.data[idx]['label']
            img_path = self.data[idx]['path']
            eye_left_encoded = self.data[idx]['eye left']
            additional_features = self.data[idx]['feature list']
            bbox = self.data[idx]['bbox']
            # Decode the eye
            eye_left_decoded = base64.b64decode(eye_left_encoded)
            eye_left_array = np.frombuffer(eye_left_decoded, dtype=np.uint8)
            eye_left = cv2.imdecode(eye_left_array, flags=cv2.IMREAD_COLOR)
            eye_left = cv2.cvtColor(eye_left, cv2.COLOR_BGR2RGB)
            if self.big_file:
                #face_encoded = self.data[idx]['face']
                nose = self.data[idx]['nose position']
                corners = self.data[idx]['corner eyes']
                #face_decoded = base64.b64decode(face_encoded)
                #face_array = np.frombuffer(face_decoded, dtype=np.uint8)
                #face = cv2.imdecode(face_array, flags=cv2.IMREAD_COLOR)
                #face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                additional_features = additional_features + nose + corners

            if self.transform:
                eye_left= self.transform(eye_left)
                #if self.big_file:
                    #face = self.transform(face)
            return  eye_left, torch.tensor(additional_features, dtype=torch.float32) , torch.tensor(label, dtype=torch.float32), torch.tensor(bbox, dtype=torch.float32), img_path  
