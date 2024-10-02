import numpy as np
import os
import cv2
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from scipy import stats

def accuracy(model, X_test, y_test):
    yhat = model.predict(X_test)

    ytrue = np.argmax(y_test, axis=1).tolist()
    yhat = np.argmax(yhat, axis=1).tolist()

    return accuracy_score(ytrue, yhat)


def get_data(train=False, test=True):
    MP_DATA = "MP_DATA" # Directory for stored numpy arrays representing frames of videos
    IMPORT_PATH = os.path.join(MP_DATA)

    flattened_actions = []
    for category in os.listdir(IMPORT_PATH):
      for action in os.listdir(os.path.join(IMPORT_PATH, category)):
        flattened_actions.append(action[3:])

    label_map = {label:num for num, label in enumerate(flattened_actions)}
    sequences, labels = [], []
    frame_range = 30

    for category in os.listdir(IMPORT_PATH):
      for action in os.listdir(os.path.join(IMPORT_PATH, category)):
        for sequence in os.listdir(os.path.join(IMPORT_PATH, category, action)):

          window = []
          for frame_no in range(frame_range):
            data = np.load(os.path.join(IMPORT_PATH, category, action, sequence, "{}.npy".format(frame_no)))
            window.append(data)

          sequences.append(window)
          labels.append(label_map[action[3:]])

    labels = np.array(labels)
    X = np.array(sequences)
    y = to_categorical(labels)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

    if train is False or test is True:
        return X_test, y_test
    elif train is True or test is False:
        return X_train, y_train


def prob_viz(res, actions, input_frame):
    if res is not None:
        output_frame = input_frame.copy()
        for num, prob in enumerate(res):
            cv2.putText(output_frame, f"{actions[num]} : {int(prob * 100)}% ", (10, 85 + num * 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2,
                        cv2.LINE_AA)
        return output_frame

    else:
        output_frame = input_frame.copy()
        for num in range(len(actions)):
            prob = 0
            cv2.putText(output_frame, f"{actions[num]} : {int(prob * 100)}% ", (10, 85 + num * 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2,
                        cv2.LINE_AA)
        return output_frame