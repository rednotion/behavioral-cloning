import os
import sys, getopt

import csv
import cv2
import numpy as np
import sklearn

from sklearn.model_selection import train_test_split
from random import shuffle
from math import ceil
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Cropping2D, Lambda, Flatten, Dense
from keras.layers import Conv2D, Dropout


# Training parameters
batch_size = 32
n_epochs = 5
model_name = 'test_model'

# Options
folder_path = '/opt/carnd_p3/data/'
data_path = 'driving_log.csv'
valid_images = ['IMG/' + x for x in  os.listdir(folder_path + 'IMG')]

variants = ["classic", "flipped", "left", "right"]
UNDER_STEER_DELTA = -0.01
OVER_STEER_DELTA = 0.02


# Parse options
try:
    opts, args = getopt.getopt(sys.argv[1:], "", ["epochs=", "name=", "batchsize="])
except getopt.GetoptError as err:
    print(err)  # will print something like "option -a not recognized"
    sys.exit(2)

for o, a in opts:
    if o == "--epochs":
        n_epochs = int(a)
        print("Setting n_epochs to {}...".format(a))
    elif o == "--batchsize":
        batch_size = int(a)
        print("Setting batch_size to {}...".format(a))
    elif o == "--name":
        model_name = a
        print("Setting model_name to {}...".format(a))

        
# Read provided data
lines = []
missing_right_images = 0
with open(folder_path + data_path) as file:
    reader = csv.reader(file)
    for l in reader:
        for v in variants:  # append one version per variant
            lines.append(l + [v, folder_path])
print("Data size = {}".format(len(lines)))

# # Data from track 2
# with open("../dom_train/driving_log.csv") as file:
#     reader = csv.reader(file)
#     for l in reader:
#         lines.append(l + ['classic', '../dom_train/'])
# print("Data size = {}".format(len(lines)))

# # Data from track 1
# track1_data = []
# with open("../liz_train/driving_log.csv") as file:
#     reader = csv.reader(file)
#     for l in reader:
#         for v in ['flipped', 'classic']:
#             track1_data.append(l + [v, '../liz_train/'])
# track1_length = len(track1_data)
# lines.extend(track1_data[0:int(0.9*track1_length)]) # use only the first 90% of data before i crashed
# print("Data size = {}".format(len(lines)))

# with open("../liz_train2/driving_log.csv") as file:
#     reader = csv.reader(file)
#     for l in reader:
#         for v in ['classic', 'flipped']:
#             lines.append(l + [v, '../liz_train2/'])
# print("Data size = {}".format(len(lines)))

# with open("../avoid_edges/driving_log.csv") as file:
#     reader = csv.reader(file)
#     for l in reader:
#         for v in ['classic', 'left_standard', 'right_standard']:
#             lines.append(l + [v, '../avoid_edges/'])
# print("Data size = {}".format(len(lines)))


# Train test split
train_samples, validation_samples = train_test_split(lines, test_size=0.33, random_state=42)
print("n train samples = {}".format(len(train_samples)))
print("n val samples = {}".format(len(validation_samples)))

    
# Get images (X) and measurements (y)
def generator(samples, batch_size=32):
    n_samples = len(samples)
    while 1:
        # shuffle(samples)
        for offset in range(0, n_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            images = []
            steering = []

            for bs in batch_samples:
                try:
                    if bs[7] == 'classic':
                        img = cv2.imread(bs[8] + bs[0])
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        images.append(img)
                        steering.append(float(bs[3]))
                    elif bs[7] in ['left', 'left_standard']:
                        img = cv2.imread(bs[8] + bs[1].replace(' ', ''))
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        images.append(img)
                        s = float(bs[3]) + UNDER_STEER_DELTA if bs[7] == 'left' else float(bs[3])
                        steering.append(s)
                    elif bs[7] in ['right', 'right_standard']:
                        img = cv2.imread(bs[8] + bs[2].replace(' ', ''))
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        images.append(img)
                        s = float(bs[3]) + OVER_STEER_DELTA if bs[7] == 'right' else float(bs[3])
                        steering.append(s)
                    elif bs[7] == 'flipped':
                        img = cv2.imread(bs[8] + bs[0])
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = np.fliplr(img)
                        images.append(img)
                        steering.append(-1 * float(bs[3]))
                    else:
                        continue
                except:
                    continue

            X_train = np.array(images)
            y_train = np.array(steering)
            yield sklearn.utils.shuffle(X_train, y_train)

            
# Generators
crop_top, crop_bottom = 40, 20
n_channels, n_rows, n_cols = 3, (160 - crop_top - crop_bottom), 320  # trimmed image
train_generator = generator(train_samples, batch_size)
validation_generator = generator(validation_samples, batch_size)

# Define model
model = Sequential()

# Preprocess model
model.add(Cropping2D(cropping=((crop_top, crop_bottom), (0, 0)), input_shape=(160, 320, 3)))
model.add(Lambda(lambda x: (x / 255.0) - 0.5,
                 input_shape=(n_rows, n_cols, n_channels),
                 output_shape=(n_rows, n_cols, n_channels)))

# Architecture
model.add(Conv2D(filters=24, kernel_size=5, strides=(2, 2), activation='relu'))
model.add(Dropout(0.5))
model.add(Conv2D(filters=36, kernel_size=5, strides=(2, 2), activation='relu'))
model.add(Dropout(0.5))
model.add(Conv2D(filters=48, kernel_size=5, strides=(2, 2), activation='relu'))
model.add(Dropout(0.5))
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
model.save('models/{}.h5'.format(model_name))
print("Saved to models/{}.h5!".format(model_name))

# Save training history as graph
fig, ax = plt.subplots()
plt.plot(training_hist.history['loss'])
plt.plot(training_hist.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig('loss_graphs/{}.png'.format(model_name))
print("Saved loss graph for {} as well".format(model_name))

