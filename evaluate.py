import os
from utils import accuracy, get_data
import tensorflow as tf
from models import load_model

MP_DATA = "MP_DATA" # Directory for stored numpy arrays representing frames of videos
IMPORT_PATH = os.path.join(MP_DATA)

flattened_actions = []
for category in os.listdir(IMPORT_PATH):
  for action in os.listdir(os.path.join(IMPORT_PATH, category)):
    flattened_actions.append(action[3:])

model = load_model('lstm_v3', flattened_actions)
print(model.summary())

X_test, y_test = get_data(train=False)
print(len(X_test))
accuracy_score = accuracy(model, X_test, y_test)
print(f"\nAccuracy : {accuracy_score}")
