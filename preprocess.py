from landmark_detection import landmark_detection_for_video, extract_landmarks, mediapipe_detection, draw_styled_landmarks
import os
import numpy as np
import mediapipe as mp
import cv2
from matplotlib import pyplot as plt

mp_holistic = mp.solutions.holistic
mp_face_mesh = mp.solutions.face_mesh
mp_drawing_utils = mp.solutions.drawing_utils

base_dir = os.path.join("E:\\dev", "ISL", "isl_detector", "isl data")

def create_folders(DATA_BASE_DIR, arrays_data_path):
  for category in os.listdir(base_dir):
    for action in os.listdir(os.path.join(base_dir, category)):
      for sequence in os.listdir(os.path.join(base_dir, category, action)):
        try:
          os.makedirs(os.path.join(arrays_data_path, category, action, sequence[:-4]))
        except:
          pass


def create_numpy_arrays(base_dir, arrays_data_path):
  with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    for action_category in os.listdir(os.path.join(base_dir)):
      for action in os.listdir(os.path.join(base_dir, action_category)):
        for sequence in os.listdir(os.path.join(base_dir, action_category, action)):
          capture = cv2.VideoCapture(os.path.join(base_dir, action_category, action, sequence))
          original_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
          original_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
          
          # Set the capture properties to maintain original dimensions
          capture.set(cv2.CAP_PROP_FRAME_WIDTH, original_width)
          capture.set(cv2.CAP_PROP_FRAME_HEIGHT, original_height)

          frame_count = 0

          while capture.isOpened() and frame_count < 60:
            ret, frame = capture.read()

            if not ret:
                break
            # Make detections
            frame = cv2.resize(frame, (original_width, original_height))

            image, results = mediapipe_detection(frame, holistic)

            keypoints = extract_landmarks(results)
            draw_styled_landmarks(image, results)
            
            '''cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Frame", original_width, original_height)
            cv2.imshow("Frame", image)'''
            
            npy_path = os.path.join(arrays_data_path, action_category, action, sequence[:-4], str(frame_count))

            # Save the detections as numpy arrays
            np.save(npy_path, keypoints)
            frame_count += 1

            if cv2.waitKey(10) & 0xFF == ord('q'):
              break
        capture.release()
        cv2.destroyAllWindows() 
        
      
create_folders(os.path.join("E:", "isl data"), 'MP_DATA')
create_numpy_arrays(base_dir, 'MP_DATA')


''' Making sure that each sequence or video is represented by 30 frames.
    This is done by deleting every other processed frame until only 30 are left
    All frames are then renamed by ascending numerical values'''

VIDEO_DATA_PATH = os.path.join("E:", "isl data")
MP_DATA_PATH = os.path.join("MP_DATA")

# Number of frames to keep
FRAMES_TO_KEEP = 30

# Loop through actions and sequences
for action_category in os.listdir(VIDEO_DATA_PATH):
  for action in os.listdir(os.path.join(VIDEO_DATA_PATH, action_category)):
    for sequence in os.listdir(os.path.join(MP_DATA_PATH, action_category, action)):
      sequence_path = os.path.join(MP_DATA_PATH, action_category, action, sequence)
      
      # List all .npy files (frames) in the sequence folder
      npy_files = [f for f in os.listdir(sequence_path) if f.endswith('.npy')]
      npy_files.sort(key=lambda x: int(x.split('.')[0]))  # Sort files by frame number
      
      # Check if there are more frames than we want to keep
      if len(npy_files) > FRAMES_TO_KEEP:
        # Use uniform sampling to select which frames to keep
        indices_to_keep = np.linspace(0, len(npy_files) - 1, FRAMES_TO_KEEP, dtype=int)
        
        # Delete the extra files
        for i, npy_file in enumerate(npy_files):
          if i not in indices_to_keep:
            file_to_delete = os.path.join(sequence_path, npy_file)
            os.remove(file_to_delete)  # Delete the file
            print(f"Deleted: {file_to_delete}")
        
        # List the remaining files again after deletion
        remaining_files = [f for f in os.listdir(sequence_path) if f.endswith('.npy')]
        remaining_files.sort(key=lambda x: int(x.split('.')[0]))  # Sort remaining files by frame number

        # Rename remaining files in ascending order
        for new_index, npy_file in enumerate(remaining_files):
          old_file_path = os.path.join(sequence_path, npy_file)
          new_file_path = os.path.join(sequence_path, f"{new_index}.npy")
          os.rename(old_file_path, new_file_path)
          print(f"Renamed: {old_file_path} -> {new_file_path}")