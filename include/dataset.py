from pickle import load
from typing_extensions import Self
from utility import*
import json
# alias
from os.path import join as join_paths

  
def read_file(file_path):
    # Returns List[dict]: A list of dictionaries representing the data.
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def write_file(output_path, data):
    # Writes a list of dictionaries to a JSON file.
    with open(output_path, 'w') as file:
        json.dump(data, file, indent=2)  # indent for pretty formatting (optional)
  
    return 

def match_in_path(path, expression):
    match = re.search(fr'{expression}(\d+)', path)
    if match:
        number = int(match.group(1))
    else:
        print("No match found")
    return number

class DataLoaderVisualizer:
    def __init__(self, root, file_path, percentage,  split='train'):
        self.root = root
        self.file_path = file_path
        self.split = split
        self.percentage = percentage
        self.path_structure = self.create_paths()
        self.drivers = self.divide_drivers()
        self.data = self.check_existence()
        Self.data_complete = []

    def divide_drivers(self):
        list_of_driver_paths = os.listdir(self.root)
        length = len(list_of_driver_paths)   
        if   self.split == 'train':
            drivers = list_of_driver_paths[:length-4]
        elif self.split == 'test':
            drivers = list_of_driver_paths[length-2:]
        elif self.split == 'val':
            drivers = list_of_driver_paths[length-4:length-2]
        else:
            raise ValueError("Invalid split. Use 'train', 'test', or 'val'.")
        return drivers

    def check_existence(self):
        if os.path.exists(self.file_path):
            print("The dataset has already been prepared, ready to use")
            data = read_file(self.file_path)
        else:
            data = self.load_data(self.percentage)  
        return data

    def create_paths(self):
        path_structure = {}
        for driver in os.listdir(self.root):
            print(f'--> Processing for {driver}')
            driver_info = {"road_view": {}, "driver_view": {}}
            # Check the driver folder exists
            driver_path = join_paths(root, driver)
            if not os.path.isdir(driver_path) or not os.listdir(driver_path):
                print(f'Skipping empty driver folder: {driver}')
                continue
            # Get paths
            driver_view_path = join_paths(root, driver, "driver_view")
            road_view_path = join_paths(root, driver, "road_view")
            # List folders
            road_sample_folders = [folder for folder in os.listdir(road_view_path) if os.path.isdir(join_paths(road_view_path, folder))]
            driver_sample_folders = [folder for folder in os.listdir(driver_view_path) if os.path.isdir(join_paths(driver_view_path, folder))]
            # Samples in road view
            for road_sample in road_sample_folders:
                print(f'--> Processing for road_sample: {road_sample}')
                sample_path = join_paths(road_view_path, road_sample)
                driver_info["road_view"][road_sample] = sample_path
            # Samples in driver view
            for driver_sample in driver_sample_folders:
                print(f'--> Processing for driver_sample: {driver_sample}')
                sample_path = join_paths(driver_view_path, driver_sample)
                driver_info["driver_view"][driver_sample] = sample_path   
            path_structure[driver] = driver_info
        #import pprint
        #pprint.pprint(path_structure)
        return path_structure
    
    def load_data(self, percentage):
      data = []
      for driver, views in self.path_structure.items():
          driver_view_samples = views['driver_view']
          data_item = {}
          data_item_complete = {}
          
          for sample_name, sample_path in driver_view_samples.items():
              video_images = [img for img in os.listdir(sample_path)]

              # Determine the number of images to load based on the specified percentage
              num_images_to_load = int(len(video_images) * (percentage / 100))
              selected_images = random.sample(video_images, num_images_to_load)
            
              
              for img in  selected_images:
                  # Save the image path
                  data_item['path'] = join_paths(sample_path, img)
                  data_item_complete[' driver path'] = join_paths(sample_path, img)
                  # Save road view path
                  data_item_complete[' road view path'] = join_paths(sample_path.replace('driver_view', 'road_view'), img)

                  # Get frame number
                  frame_number = int(img.split("_")[1].split(".")[0])
                  gt_path = sample_path.replace('driver_view', 'road_view') + '.npy'
                  gt_array = np.load(gt_path, allow_pickle=True)
                  sample_number = match_in_path(gt_path, sample_name)
                  frame_number_label, gaze_point, boundbox = get_frame_and_point(gt_array, frame_number, sample_number)      
                  assert frame_number == frame_number_label, f"Not the same frame number: frame_number={frame_number}, " \
                                                              f"frame_number_label={frame_number_label}, img_path ={img}, " \
                                                              f"array={gt_path}" 
                  # Save Lavel
                  data_item['label']  = list(gaze_point)
                  data_item_complete['label']  = list(gaze_point)             
      return data


    def _visualize_dataset(self):
        for driver,_ in self.path_structure.items():
          driver_x_paths = find_driver_paths(self.data_complete['paths'], driver)
          driver_img = random.sanple(driver_x_paths)
          road_image = driver_img

        
        road_photo = cv2.imread(road_photo_path)
        road_photo = cv2.cvtColor(road_photo, cv2.COLOR_BGR2RGB)

        # Bounding box from .npy files
        rect = patches.Rectangle((boundbox[0], boundbox[1]), boundbox[2]-boundbox[0], boundbox[3]-boundbox[1],
                                 linewidth=1, edgecolor='r', facecolor='none')

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

   

class DGAZEDataset(Dataset):
    def __init__(self, loader_class, split='train', save_file='train_paths.json'):
        self.split = split
        
        if match_in_path(save_file, split):
            self.save_file = save_file
        else:
            raise ValueError("You used the wrong path")

        self.loader_class = loader_class
        self.data = self.loader_class.load_data(save_file)  

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

        return image, torch.tensor(label, dtype=torch.float32)
