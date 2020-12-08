import cv2
import numpy as np
import sklearn

from keras.models import Sequential
from keras.layers import Cropping2D

data_path = '/opt/carnd_p3/data/'

# Read file
lines = []
with open("../data/driving_log.csv") as file:
    reader = csv.reader(file)
    for l in reader:
        lines.append(l)

# Get images (X) and measurements (y)
def generator(samples, batch_size=32):
    n_samples = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0, n_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            
            images = []
            steering = []
            
            for bs in batch_samples:
                center_img = cv2.imread(bs[0])
                # left_img = cv2.imread(bs[1])
                # right_img = cv2.imread(bs[2])
                images.append(cv2.imread(center_img))
                # image_flipped = np.fliplr(image)ls 
                steering.append(float(bs[3]))
            
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
model.add(Cropping2D(cropping=((crop_top,crop_bottom), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: x/127.5 - 1.,
                 input_shape=(n_channels, n_rows, n_cols),
                 output_shape=(n_channels, n_rows, n_cols)))

# Architecture

# Compile and train
model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator,
                    steps_per_epoch=ceil(len(train_samples)/batch_size),
                    validation_data=validation_generator,
                    validation_steps=ceil(len(validation_samples)/batch_size),
                    epochs=n_epochs,
                    verbose=1)

# Save
model.save('model.h5')                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                