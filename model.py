import csv
import cv2
import numpy as np

DATA_DIR = 'data/'
IMG_DIR = 'IMG/'
images = []
measurements = []
correction = 0.2
with open(DATA_DIR + 'driving_log.csv', 'rt') as csvfile:
  reader = csv.reader(csvfile, delimiter=',', quotechar='|')
  for line in reader:
    for i in range(3): # 0: center 1: left 2: right
      source_path = line[i]
      tokens = source_path.split('/')
      filename = tokens[-1]
      local_path = DATA_DIR + IMG_DIR + filename;
 #     print(local_path)
      image = cv2.imread(local_path)
#      print(image)
      images.append(image)
      flipped_image = cv2.flip(image, 1)
      images.append(flipped_image)
#    print(images)
#    exit();
    measurement = float(line[3])
    measurements.append(measurement)
    measurements.append(measurement+correction) # left
    measurements.append(measurement-correction) # right
    flipped_measurement = measurement * -1.0
    measurements.append(flipped_measurement)
    measurements.append(flipped_measurement+correction)
    measurements.append(flipped_measurement-correction)

X_train = np.array(images)
y_train = np.array(measurements)
print(X_train.shape)

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
model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
#model.add(MaxPooling2D())
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))
#model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))

model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

model.summary()

model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)

model.save('model.h5')
