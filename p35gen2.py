import os
import csv

lines = []
with open('./driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

#print (lines[-2])
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

print ('# of training/Val samples: ', len(train_samples), len(validation_samples))

import cv2
import numpy as np
import sklearn

from keras.models import Sequential
from keras.layers.convolutional import Cropping2D 
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers import Convolution2D, MaxPooling2D, Cropping2D
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        #shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            #For Unix style path, use '/' for split() below, else, use '\\' for DOS style paths
            for batch_sample in batch_samples:
                fname = batch_sample[0].split('/')[-1]
                name = './IMG/'+ fname
                center_image = mpimg.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
                
                fname = batch_sample[1].split('/')[-1]
                name = './IMG/'+ fname
                left_image = mpimg.imread(name)
                left_angle = center_angle + 0.15
                images.append(left_image)
                angles.append(left_angle)
                #Flipping to remove turn bias
                images.append(cv2.flip(left_image, 1))
                angles.append(left_angle*-1.0)
                
                fname = batch_sample[2].split('/')[-1]
                name = './IMG/'+ fname
                right_image =  mpimg.imread(name)
                right_angle =  center_angle - 0.15
                images.append(right_image)
                angles.append(right_angle)
                #Flipping to remove turn bias
                images.append(cv2.flip(right_image, 1))
                angles.append(right_angle*-1.0)
                
            # trim image to only see section with road
            #print ('Length of images in generator:', len(images), len(angles))
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=16)
validation_generator = generator(validation_samples, batch_size=16)

#ch, row, col = 3, 80, 320  # Trimmed image format

model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation 
#model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(ch, row, col), output_shape=(ch, row, col)))
model.add (Lambda(lambda x: x/255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((30,20), (0,0)), input_shape=(160, 320, 3)))
#model.add(... finish defining the rest of your model architecture here ...)
model.add(Convolution2D(24,(5,5), activation="relu", name="conv1"))
model.add(MaxPooling2D())

model.add(Convolution2D(36, (5,5), activation="relu", name="conv2"))
model.add(MaxPooling2D())

model.add(Convolution2D(48, (5,5), activation="relu", name="conv3"))
model.add(MaxPooling2D())

model.add(Convolution2D(64, (3,3), activation="relu", name="conv4"))
model.add(MaxPooling2D())

model.add(Convolution2D(64, (3,3), activation="relu", name="conv5"))
model.add(MaxPooling2D())
model.add(Flatten())

model.add(Dense(1164))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

#for i in range(18):
#	print (model.layers[i].output)

model.compile(loss='mse', optimizer='adam')
#model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=3) Keras 1.2
history = model.fit_generator(train_generator, validation_steps=len(validation_samples), epochs=3, validation_data=validation_generator, steps_per_epoch=len(train_samples)) #Keras 2.0

import sys
fname = sys.argv[0]
model.save((fname.replace(".py", ".h5")))
#model.summary()

#del history
#del model
#gc.collect()

t3 = time.time()
print ('Finished training/val data in ', round(t3-t2,2), ' sec. \nTotal Elapsed time: ', round(t3-t1, 2), ' sec.')