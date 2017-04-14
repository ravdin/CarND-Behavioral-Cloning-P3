import csv
import cv2
import numpy as np

nb_epoch = 2
correction = 0.1
image_paths, angles = [], []
with open('../data/driving_log.csv') as csvfile:
  reader = csv.reader(csvfile)
  for row in reader:
    if float(row[6]) < 0.1:
      continue
    image_paths.extend(row[0:3])
    steering_center = float(row[3])
    steering_left = steering_center + correction
    steering_right = steering_center - correction
    angles.extend([steering_center, steering_left, steering_right])

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
image_paths_train, image_paths_test, angles_train, angles_test = train_test_split(image_paths, angles, test_size=0.2)

def preprocessImage(image, angle):
    # Image is 320x160x3
    # Crop to 250x160x3
    image = image[0:160, 50:300, :]
    image = augment_brightness_camera_images(image)
    # Resize to 200x66x3
    image = cv2.resize(image, (200, 66))
    #Randomly flip half the images on a horizontal axis
    if np.random.random_sample() < 0.5:
        image = cv2.flip(image, 1)
        angle = -angle
    return (image, angle)

# Source for this method:
# https://medium.com/@vivek.yadav/improved-performance-of-deep-learning-neural-network-models-on-traffic-sign-classification-using-6355346da2dc
def augment_brightness_camera_images(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    image1 = np.array(image1, dtype = np.float64)
    random_bright = .5+np.random.uniform()
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1[:,:,2][image1[:,:,2]>255]  = 255
    image1 = np.array(image1, dtype = np.uint8)
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1

def generator(image_paths, angles, batch_size=32):
  num_samples = len(image_paths)
  X, y = [], []
  image_paths, angles = shuffle(image_paths, angles)
  while 1:
    for i in range(batch_size):
      (image, angle) = preprocessImage(cv2.imread(image_paths[i]), angles[i])
    X.append(image)
    y.append(angle)
    if len(X) == batch_size:
        yield (np.array(X), np.array(y))
        X, y = [], []
        image_paths, angles = shuffle(image_paths, angles)

train_generator = generator(image_paths_train, angles_train, batch_size=32)
validation_generator = generator(image_paths_test, angles_test, batch_size=32)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: x / 127.5 - 1., input_shape=(66,200,3), output_shape=(66,200,3)))
model.add(Convolution2D(24,(5,5),strides=(2,2),activation="relu"))
model.add(Convolution2D(36,(5,5),strides=(2,2),activation="relu"))
model.add(Convolution2D(48,(5,5),strides=(2,2),activation="relu"))
model.add(Convolution2D(64,(3,3),activation="relu"))
model.add(Convolution2D(64,(3,3),activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam', lr=1e-3)
model.fit_generator(train_generator, steps_per_epoch=len(angles_train), validation_data=validation_generator, validation_steps=len(angles_test), epochs=nb_epoch)

model.save('model.h5')
