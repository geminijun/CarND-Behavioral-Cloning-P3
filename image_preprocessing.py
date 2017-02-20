import csv
import matplotlib.pyplot as plt
import cv2

DATA_DIR = 'data/'
IMG_DIR = 'IMG/'
corrections = [0, 0.2, -0.2]

with open(DATA_DIR + 'driving_log.csv', 'r') as csvfile:
  reader = csv.reader(csvfile, delimiter=',')
  fig = plt.figure()
  img_num = 0
  for line in reader:
    measurement = float(line[3])
    for i in range(3): # 0: center 1: left 2: right
      source_path = line[i]
      tokens = source_path.split('/')
      filename = tokens[-1]
      local_path = DATA_DIR + IMG_DIR + filename;
      image = cv2.imread(local_path)

      fig.add_subplot(3, 3, img_num * 3 + i + 1)
      plt.title(str(round(measurement+corrections[i], 3)))
      plt.imshow(image)
    img_num = img_num+1
    if img_num >= 3:
      break

plt.show()

#      flipped_image = cv2.flip(image, 1)
#    print(images)
#    exit();
#    measurements.append(measurement)
#    measurements.append(measurement+correction) # left
#    measurements.append(measurement-correction) # right
#    flipped_measurement = measurement * -1.0
#    measurements.append(flipped_measurement)
#    measurements.append(flipped_measurement+correction)
#    measurements.append(flipped_measurement-correction)
