import os
import cv2

def delete_augmented_videos(base_dir):
    """
    Deletes all augmented videos (files starting with 'augmented_') from the specified base directory.
    """
    for category in os.listdir(base_dir):
      category_path = os.path.join(base_dir, category)
      if os.path.isdir(category_path):
        for sign in os.listdir(category_path):
          sign_path = os.path.join(category_path, sign)
          if os.path.isdir(sign_path):
            # Loop through the files in the sign directory
            for file in os.listdir(sign_path):
              if file.startswith('augmented_') and file.endswith('.mov'):
                file_path = os.path.join(sign_path, file)
                os.remove(file_path)
                print(f"Deleted: {file_path}")

import random

def augment_video(video_path, output_path, augmentation_type):
    """
    Apply video augmentation and save the result.
    augmentation_type: ('rotate', 'flip', 'speed').
    """

    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # FourCC for saving .mov
    out = None
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if augmentation_type == 'speed':
        new_fps = fps * random.uniform(0.75, 1.25)  # Slight speed change
    else:
        new_fps = fps
    
    out = cv2.VideoWriter(output_path, fourcc, new_fps, (frame_width, frame_height))
    
    # Apply the same rotation to all frames if augmentation is 'rotate'
    if augmentation_type == 'rotate':
        angle = random.choice([15, 30, -15, -30])  # Choose a small consistent angle
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if augmentation_type == 'rotate':
            center = (frame.shape[1] // 2, frame.shape[0] // 2)
            matrix = cv2.getRotationMatrix2D(center, angle, 1)  # Use consistent angle for all frames
            frame = cv2.warpAffine(frame, matrix, (frame.shape[1], frame.shape[0]))
        
        elif augmentation_type == 'flip':
            frame = cv2.flip(frame, 1)  # Horizontal flip
        
        out.write(frame)
    
    cap.release()
    out.release()

def augment_videos_in_directory(base_dir, target_count=21):
  """
  Augment videos for each sign in the directory until each sign has at least target_count videos.
  """
  augmentations = ['rotate', 'flip', 'speed']
  
  for category in os.listdir(base_dir):
    category_path = os.path.join(base_dir, category)
    if os.path.isdir(category_path):
      for sign in os.listdir(category_path):
        sign_path = os.path.join(category_path, sign)
        if os.path.isdir(sign_path):
          video_files = [f for f in os.listdir(sign_path) if f.endswith('.MOV')]
          video_count = len(video_files)
          
          # Skip if no videos are found
          if video_count == 0:
            print(video_files)
            print(f"No videos found for sign: {sign}")
            continue
          
          if video_count < target_count:
            for i in range(video_count, target_count):
              original_video = random.choice(video_files)
              augmentation_type = random.choice(augmentations)
              output_video = os.path.join(sign_path, f"augmented_{i}_{augmentation_type}.mov")
              
              augment_video(os.path.join(sign_path, original_video), output_video, augmentation_type)
              print(f"Augmented video created: {output_video}")


base_directory = os.path.join('E:', 'isl data')
augment_videos_in_directory(base_directory, target_count=30)
