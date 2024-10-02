import mediapipe as mp
import cv2
import numpy as np
import os
import time

mp_holistic = mp.solutions.holistic
mp_face_mesh = mp.solutions.face_mesh
mp_drawing_utils = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
  image.flags.writeable = False                   # Image is no longer writeable
  results = model.process(image)                 # Make prediction
  image.flags.writeable = True                   # Image is now writeable 
  image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
  return image, results

def draw_landmarks(image, results):
  mp_drawing_utils.draw_landmarks(image, results.face_landmarks, mp_face_mesh.FACEMESH_TESSELATION) # Draw face connections
  mp_drawing_utils.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Draw pose connections
  mp_drawing_utils.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
  mp_drawing_utils.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand connections

def draw_styled_landmarks(image, results):
    # Draw face connections
    mp_drawing_utils.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
                             mp_drawing_utils.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
                             mp_drawing_utils.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                             ) 
    # Draw pose connections
    mp_drawing_utils.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing_utils.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing_utils.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             ) 
    # Draw left hand connections
    mp_drawing_utils.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing_utils.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing_utils.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
    # Draw right hand connections  
    mp_drawing_utils.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing_utils.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing_utils.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             )


def extract_landmarks(results):
  pose_landmarks = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
  face_landmarks = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
  left_hand_landmarks = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
  right_hand_landmarks = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)

  return np.concatenate([pose_landmarks, face_landmarks, left_hand_landmarks, right_hand_landmarks])


def landmark_detection_for_video(video_path, model):
  sequence = []
  cap = cv2.VideoCapture(video_path)

  with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
      ret, frame = cap.read()
      temp = frame

      if not ret:
        break
      # Make detections
      frame = cv2.resize(frame, (640, 480))
      image, results = mediapipe_detection(frame, holistic)
      sequence.append(results)

      # Break gracefully
      if cv2.waitKey(10) & 0xFF == ord('q'):
        break

    cap.release()
    cv2.destroyAllWindows()
    return sequence

