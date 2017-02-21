import csv
import cv2
import numpy as np

DATA_DIR = 'data/'
IMG_DIR = 'IMG/'
correction = 0.25
samples = []
with open(DATA_DIR + 'driving_log.csv', 'r') as csvfile:
  reader = csv.reader(csvfile, delimiter=',')
  for line in reader:
    samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import sklearn

def generator(samples, batch_size = 128):
  num_samples = len(samples)
  while 1:
    sklearn.utils.shuffle(samples)
    for offset in range(0, num_samples, batch_size):
      batch_samples = samples[offset:offset+batch_size]
      images = []
      measurements = []
      for batch_sample in batch_samples:
        for i in range(3): # 0: center 1: left 2: right
          source_path = batch_sample[i]
          tokens = source_path.split('/')
          filename = tokens[-1]
          local_path = DATA_DIR + IMG_DIR + filename;
     #     print(local_path)
          image = cv2.imread(local_path)
    #      print(image)
          images.append(image)
          flipped_image = cv2.flip(image, 1)
          images.append(flipped_image)
        measurement = float(line[3])
        flipped_measurement = measurement * -1.0
        measurements.append(measurement)
        measurements.append(flipped_measurement)
        measurements.append(measurement+correction) # left
        measurements.append(flipped_measurement+correction)
        measurements.append(measurement-correction) # right
        measurements.append(flipped_measurement-correction)

      X_train = np.array(images)
      y_train = np.array(measurements)
      yield sklearn.utils.shuffle(X_train, y_train)

train_generator = generator(train_samples, batch_size=128)
validation_generator = generator(validation_samples, batch_size=128)

import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: x/255. - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((60, 20), (0,0))))
model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
#model.add(MaxPooling2D())
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))
#model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))

model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

model.summary()

#model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=6)

history_object = model.fit_generator(train_generator, samples_per_epoch = len(train_samples)*6, validation_data = validation_generator, nb_val_samples = len(validation_samples)*6, nb_epoch=6, verbose=1)

### print the keys contained in the history object
print(history_object.history.keys())

# import matplotlib.pyplot as plt

# ### plot the training and validation loss for each epoch
# plt.plot(history_object.history['loss'])
# plt.plot(history_object.history['val_loss'])
# plt.title('model mean squared error loss')
# plt.ylabel('mean squared error loss')
# plt.xlabel('epoch')
# plt.legend(['training set', 'validation set'], loc='upper right')
# plt.show()


model.save('model.h5')
