import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization

def compile_model(model):
  adam = tf.keras.optimizers.Adam(3e-4)
  model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
  return model

def lstm_v3(device_name, len_output):

  with tf.device(device_name):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=(30, 1662)))
    model.add(LSTM(256, return_sequences=True))
    model.add(LSTM(128, return_sequences=False))

    model.add(Dense(1024, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(len_output, activation='softmax'))

    return model

'''def lstm_v3(device_name, actions):

  with tf.device(device_name):
    model = Sequential()

    model.add(LSTM(128, return_sequences=True, input_shape=(30, 1662)))  # 30 timesteps, 1662 features
    model.add(Dropout(0.5))

    model.add(LSTM(128, return_sequences=True))  # Another LSTM layer to capture more temporal patterns
    model.add(Dropout(0.5))

    model.add(LSTM(64, return_sequences=False))  # Final LSTM layer
    model.add(Dropout(0.5))

    # Dense layers
    model.add(Dense(128, activation='relu'))  # First dense layer
    model.add(BatchNormalization())  # Helps stabilize training
    model.add(Dropout(0.5))

    model.add(Dense(64, activation='relu'))  # Second dense layer
    model.add(Dropout(0.5))

    # Output layer for classification (9 signs)
    model.add(Dense(len(actions), activation='softmax'))

    return model'''

def load_model(name='lstm_v3', pretrained=False, training=True, device=None, len_output = 0):
  if not pretrained:
    model = lstm_v3(device, len_output)
    if training:
      return compile_model(model)
    else:
      return model
    
  model_dir = os.path.join('models')

  model_path = [os.path.join(model_dir, _) for _ in os.listdir(model_dir) if _.endswith(r".keras")][0]
  print(f"Loading Model from : {model_path}")

  model = tf.keras.models.load_model(model_path)
  return model
    