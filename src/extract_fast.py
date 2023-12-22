import os
drive_folder = '/home/anto/University/Driving-Visual-Attention/data'
#drive_folder = os.getcwd()

import subprocess
from tqdm import tqdm
import shutil

def extract_frames(video_path, output_folder):
    os.makedirs(output_folder) #exist_ok=True

    # Use FFmpeg to extract frames
    command = [
        'ffmpeg',
        '-analyzeduration', '100M',  # Aumenta il valore di analyzeduration
        '-probesize', '100M',        # Aumenta il valore di probesize
        '-i', video_path,
        os.path.join(output_folder, 'frame_%04d.jpg')
    ]

    result= subprocess.run(command, stderr=subprocess.PIPE)

    if result.returncode != 0:
      print(f"Errore durante l'estrazione dei frame:\n{result.stderr.decode('utf-8')}")

    return

import cv2

#ESTRAI FRAME CON CV2
def extract_frames_with_cv2(video_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    # Apre il video
    cap = cv2.VideoCapture(video_path)

    # Contatore per numerare i frame
    count = 0

    while cap.isOpened():
        # Legge un frame
        ret, frame = cap.read()

        if ret:
            # Salva il frame come immagine
            frame_filename = os.path.join(output_folder, f'frame_{count:04d}.jpg')
            success=cv2.imwrite(frame_filename, frame)
            if not success:
              print(f"Errore nel salvataggio dell'immagine {frame_filename}")

            count += 1
        else:
            break

    # Rilascia il video
    cap.release()

    return

def process_dataset(drive_folder, new_folder, dataset_type='videos_aligned'):

    for driver in os.listdir(os.path.join(drive_folder, dataset_type)):

        driver_folder_path = os.path.join(drive_folder, new_folder, driver)  # Path to the driver folder in new_folder
        if os.path.exists(driver_folder_path):  # Check if the driver folder has been created
            print(f"Skipping {driver}, folder already exists.")
            continue

        print(f'--> Processing for {driver}')
        nsamples = len(os.listdir(os.path.join(drive_folder, dataset_type, driver, 'driver_view')))
        nsamples_range = list(range(1, nsamples + 1))
        for sample_no in tqdm(nsamples_range):
            driver_view_path = os.path.join(drive_folder, dataset_type, driver, 'driver_view', 'sample' + str(sample_no) + '.avi')
            road_view_path = os.path.join(drive_folder, dataset_type, driver, 'road_view', 'sample' + str(sample_no) + '.avi')
            gaze_point_path = os.path.join(drive_folder, dataset_type, driver, 'road_view', 'sample' + str(sample_no) + '.npy')
            extract_frames_with_cv2(driver_view_path, os.path.join(drive_folder,new_folder,driver, 'driver_view', 'sample' + str(sample_no)))
            extract_frames_with_cv2(road_view_path, os.path.join(drive_folder, new_folder,driver, 'road_view', 'sample' + str(sample_no)))
            shutil.copy(gaze_point_path, os.path.join(drive_folder, new_folder,driver, 'road_view', 'sample' + str(sample_no) + '.npy'))

    print("Processing completed.")

process_dataset(drive_folder , new_folder = 'images_aligned')