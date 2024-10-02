import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from models import load_model

MP_DATA = "MP_DATA" # Directory for stored numpy arrays representing frames of videos
device = 'gpu:0' if tf.test.is_gpu_available() else 'cpu' # Use GPU for trainng if possible 
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

print(labels)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

print(len(flattened_actions))

model = load_model('lstm_v3', pretrained=False, training=True, device=device, actions = flattened_actions)
callbacks = [

  tf.keras.callbacks.ModelCheckpoint("isl_model.keras", save_best_only=True),

  tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                        factor=0.1,
                                        patience=10,
                                        verbose=0,
                                        mode='auto',
                                        min_delta=0.0001,
                                        cooldown=15,
                                        min_lr=0),

  # tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)
]

with tf.device(device):
  history = model.fit(X_train, y_train,
                      epochs=500,
                      validation_data=(X_test, y_test),
                      callbacks=callbacks)

