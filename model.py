import csv
import cv2
import numpy as np

nb_epoch = 35
correction = 0.25
data = []
with open('../data/driving_log.csv') as csvfile:
  reader = csv.reader(csvfile)
  for row in reader:
    # Skip samples where the speed is low.
    if float(row[6]) < 1:
      continue
    image_center = row[0]
    steering_center = float(row[3])
    # Reject 95% of the samples where the steering angle is 0.
    if steering_center == 0 and np.random.random() > 0.05:
        continue
    data.append((image_center, steering_center, False))
    if steering_center != 0:
        data.append((image_center, -steering_center, True))
    # For a right turn, use the left camera image and add correction
    if steering_center > 0 and steering_center + correction <= 1.0:
        image_left = row[1]
        angle = steering_center + correction
        data.append((image_left, angle, False))
        data.append((image_left, -angle, True))
    # For a left turn, use the right camera image and subtract correction
    if steering_center < 0 and steering_center - correction >= -1.0:
        image_right = row[2]
        angle = steering_center - correction
        data.append((image_right, angle, False))
        data.append((image_right, -angle, True))

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
data = shuffle(data)
data_train, data_test = train_test_split(data, test_size=0.05)

def preprocessImage(image, flip):
    # Image is 320x160x3
    # Crop to 250x160x3
    result = image[0:160, 50:300, :]
    result = cv2.GaussianBlur(result, (5, 5), 0)
    # Resize to 200x66x3
    result = cv2.resize(result, (200, 66), interpolation = cv2.INTER_AREA)
    if flip:
        result = cv2.flip(result, 1)
    return result

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

def generator(items, batch_size=32, validation_flag=False):
    num_samples = len(items)
    X, y = [], []
    items = shuffle(items)
    while 1:
        for i in range(batch_size):
           (image_path, angle, flip) = items[i]
           image = preprocessImage(cv2.imread(image_path), flip)
           if not validation_flag:
               image = augment_brightness_camera_images(image)
           X.append(image)
           y.append(angle)
        yield (np.array(X), np.array(y))
        X, y = [], []
        items = shuffle(items)

batch_size = 128
num_steps = len(data_train) / batch_size
train_generator = generator(data_train, batch_size, False)
validation_generator = generator(data_train, batch_size, True)
test_generator = generator(data_test, batch_size, True)

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
model.fit_generator(train_generator, steps_per_epoch=num_steps, validation_data=validation_generator, validation_steps=num_steps, epochs=nb_epoch)

model.save('model.h5')

for X_test, y_test in test_generator:
    test_loss = model.test_on_batch(X_test, y_test)
    print('Test loss: {0}'.format(test_loss))
    break
