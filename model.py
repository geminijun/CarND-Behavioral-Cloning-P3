import csv
import cv2

DATA_DIR = 'data/'
lines = []
with open(DATA_DIR + 'driving_log.csv', 'rb') as csvfile:
  reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
  for line in reader:
    lines.append(line)
images = []
measurements = []
for line in lines:
  for i in range(3):
    source_path = line[i]
    tokens = source_path.split('/')
    filename = tokens[-1]
    local_path = "./data/" + filename;
    image = cv2.imread(local_path)
    images.append(image)
  correction = 0.2
  measurement = float(line[3])
  measurements.append(measurement)
  measurements.append(measurement+correction)
  measurements.append(measurement-correction)

augmented_images = []
augmented_measurements = []
for image, measurement in zip(images, measurements):
  augmented_measurements.append(measurement)
  augmented_images.append(image)
  flipped_image = cv2.flip(image, 1)
  flipped_measurement = measurement * -1.0
  augmented_images.append(flipped_image)
  augmented_measurements.append(flipped_measurement)

import numpy as np

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

# Python generator

# 5k images


import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: x/255. - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70, 25), (0,0))))
model.add(Convolution2D(6,5,5,activation='relu'))
model.add(MaxPooling2D())
model.add(Convolution2D(16,5,5,activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)

model.save('model.h5')
