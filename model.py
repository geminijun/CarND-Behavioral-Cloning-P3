import csv
import cv2
import numpy as np

DATA_DIR = 'data/'
IMG_DIR = 'IMG/'
corrections = [0, 0.25, -0.25]
samples = []
with open(DATA_DIR + 'driving_log.csv', 'r') as csvfile:
  reader = csv.reader(csvfile, delimiter=',')
  for line in reader:
    measurement = float(line[3])
    for i in range(3): # 0: center 1: left 2: right
      sample = {}
      source_path = line[i]
      tokens = source_path.split('/')
      filename = tokens[-1]
      local_path = DATA_DIR + IMG_DIR + filename;
      sample['path'] = local_path
      sample['flipped'] = False
      sample['angle'] = measurement + corrections[i]
      samples.append(sample)
      sample = {}
      sample['path'] = local_path
      sample['flipped'] = True
      sample['angle'] = (measurement + corrections[i]) * (-1.0)
      samples.append(sample)

print(len(samples))

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
samples = shuffle(samples)
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def generator(samples, batch_size = 128):
  num_samples = len(samples)
  while 1:
    samples = shuffle(samples)
    for offset in range(0, num_samples, batch_size):
      batch_samples = samples[offset:offset+batch_size]
      images = []
      measurements = []
      for batch_sample in batch_samples:
        image = cv2.imread(batch_sample['path'])
        if batch_sample['flipped']:
          image = cv2.flip(image, 1)
        images.append(image)
        measurements.append(batch_sample['angle'])

      X_train = np.array(images)
      y_train = np.array(measurements)
      yield shuffle(X_train, y_train)

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

history_object = model.fit_generator(train_generator, samples_per_epoch = len(train_samples), validation_data = validation_generator, nb_val_samples = len(validation_samples), nb_epoch=5, verbose=1)

### print the keys contained in the history object
print(history_object.history.keys())

import matplotlib.pyplot as plt

# ### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()


model.save('model.h5')
