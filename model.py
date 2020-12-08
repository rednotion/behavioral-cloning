import csv
import cv2
import numpy as np
import sklearn

from random import shuffle
from math import ceil
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Cropping2D, Lambda, Flatten, Dense
from keras.layers import Conv2D

model_name = 'test_model'

data_path = '/opt/carnd_p3/data/'

variants = ['classic']  # 'left', 'right', 'flipped'
UNDER_STEER_DELTA = 0.05
OVER_STREER_DELTA = -0.05

# Read file
lines = []
with open("../data/driving_log.csv") as file:
    reader = csv.reader(file)
    for l in reader:
        # append one version per variant
        for v in variants:
            lines.append(l + [v])

# Train test split
train_samples, validation_samples = sklearn.model_selection.train_test_split(lines, test_size=0.33, random_state=42)


# Get images (X) and measurements (y)
def generator(samples, batch_size=32):
    n_samples = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0, n_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            images = []
            steering = []

            for bs in batch_samples:
                if bs[4] == 'classic':
                    images.append(cv2.imread(bs[0]))
                    steering.append(float(bs[3]))
                elif bs[4] == 'left':
                    images.append(cv2.imread(bs[1]))
                    steering.append(float(bs[3]) + UNDER_STEER_DELTA)
                elif bs[4] == 'right':
                    images.append(cv2.imread(bs[2]))
                    steering.append(float(bs[3]) + OVER_STREER_DELTA)
                elif bs[4] == 'flipped':
                    img = cv2.imread(bs[0])
                    img = np.fliplr(img)
                    images.append(img)
                    steering.append(-float(bs[3]))
                else:
                    pass

            X_train = np.array(images)
            y_train = np.array(steering)
            yield sklearn.utils.shuffle(X_train, y_train)


# Training parameters
n_epochs = 5
batch_size = 32
learning_rate = 0.005
crop_top, crop_bottom = 40, 20
n_channels, n_rows, n_cols = 3, (160 - crop_top - crop_bottom), 320  # trimmed image

# Generators
train_generator = generator(train_samples, batch_size)
validation_generator = generator(validation_samples, batch_size)

# Define model
model = Sequential()

# Preprocess model
model.add(Cropping2D(cropping=((crop_top, crop_bottom), (0, 0)), input_shape=(160, 320, 3)))
model.add(Lambda(lambda x: (x / 255.0) - 0.5,
                 input_shape=(n_channels, n_rows, n_cols),
                 output_shape=(n_channels, n_rows, n_cols)))

# Architecture
model.add(Conv2D(filters=24, kernel_size=5, strides=(2, 2), activation='relu'))
model.add(Conv2D(filters=36, kernel_size=5, strides=(2, 2), activation='relu'))
model.add(Conv2D(filters=48, kernel_size=5, strides=(2, 2), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=3, strides=(1, 1), activation='relu'))
model.add(Flatten())  # 512
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=56, activation='relu'))
model.add(Dense(units=10, activation='relu'))
model.add(Dense(units=1))

# Compile and train
model.compile(loss='mse', optimizer='adam')
training_hist = model.fit_generator(train_generator,
                                    steps_per_epoch=ceil(len(train_samples) / batch_size),
                                    validation_data=validation_generator,
                                    validation_steps=ceil(len(validation_samples) / batch_size),
                                    epochs=n_epochs,
                                    verbose=1)

# Save
model.save('{}.h5'.format(model_name))

# Save training history as graph
fig, ax = plt.subplots()
plt.plot(training_hist.history['loss'])
plt.plot(training_hist.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig('{}_loss_progress.png'.format(model_name))
