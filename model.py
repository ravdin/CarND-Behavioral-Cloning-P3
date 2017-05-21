import csv
import cv2
import numpy as np

nb_epoch = 5
correction = 0.25
data = []
with open('../data/driving_log.csv') as csvfile:
  reader = csv.reader(csvfile)
  for row in reader:
    image_center = row[0]
    steering_center = float(row[3])
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
data_train, data_valid = train_test_split(data, test_size=0.1)

def preprocessImage(image, flip):
    cropped = image[60:140,:]
    result = cv2.resize(cropped, (64, 64))
    if flip:
        result = cv2.flip(result, 1)
    return result

def augment_brightness_camera_images(image):
    hsv = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = np.random.uniform(0.3, 1.0)
    hsv[:,:,2] = hsv[:,:,2]*random_bright
    return cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)

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

input_shape = (64, 64, 3)
batch_size = 128
train_steps = len(data_train)
validation_steps = len(data_valid)
train_generator = generator(data_train, batch_size, False)
validation_generator = generator(data_valid, batch_size, True)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam
from keras.regularizers import l2

model = Sequential()
model.add(Lambda(lambda x: x / 255 - 0.5, input_shape=input_shape))
model.add(Convolution2D(24,(5,5),strides=(2,2),activation="relu",kernel_regularizer=l2(0.001)))
model.add(Convolution2D(36,(5,5),strides=(2,2),activation="relu",kernel_regularizer=l2(0.001)))
model.add(Convolution2D(48,(5,5),strides=(2,2),activation="relu",kernel_regularizer=l2(0.001)))
model.add(Convolution2D(64,(3,3),activation="relu",kernel_regularizer=l2(0.001)))
model.add(Convolution2D(64,(3,3),activation="relu",kernel_regularizer=l2(0.001)))
model.add(Flatten())
model.add(Dense(100,kernel_regularizer=l2(0.001)))
model.add(Dropout(0.5))
model.add(Dense(50,kernel_regularizer=l2(0.001)))
model.add(Dropout(0.5))
model.add(Dense(10,kernel_regularizer=l2(0.001)))
model.add(Dropout(0.5))
model.add(Dense(1,kernel_regularizer=l2(0.001)))

adam = Adam(lr = 1e-3)
model.compile(optimizer= adam, loss='mse', metrics=['accuracy'])
model.fit_generator(train_generator, steps_per_epoch=train_steps, validation_data=validation_generator, validation_steps=validation_steps, epochs=nb_epoch)

model.save('model.h5')
