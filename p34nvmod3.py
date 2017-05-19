#s
#Vijay D. Udacity CarND: Project 3

#p34nvmod2.py: This is the Nvidia pipeline for P3, becoz of memory limitations in AWS, I added MaxPooling2D() layers.
#p34nvmod3.py: Added dropout layers (noted slight erformance decrease but better driving!)

import numpy as np  
import cv2
import csv

lines = []

with open('./driving_log.csv') as csvfile:
	reader = csv.reader (csvfile)
	for line in reader:
		lines.append(line)

images = []
measurements = []
for line in lines[1:]:
	source_path = line[0] 
	filename = source_path.split('\\')[-1]
	#print ('SP', source_path, filename)
	current_path = 'IMG\\' + filename
	#print ('CP', current_path)
	image = cv2.imread(current_path)
	#print (image)
	images.append(image)
	
	measurement = float(line[3])
	measurements.append(measurement)
	
print (len(images))	
# Flipping to remove the turn bias, and also crop img to rid it of the useless pixels	top 70 pixels and btm 20 pixels, retain entire width
# Crop from x, y, w, h -> 0:0 origin, 320, 400
# NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]

aug_img, aug_meas = [], []
for img, meas, in zip (images, measurements):
	#First shrink the size of image by half
	#img = cv2.resize(img, (0,0), fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
	#Crop first before appending
	#img = img[70:140, 1:320] #Rmv top 70px, btm 20pix, w=320, so ygoes from 69:139 px, X goes from 0:319
	aug_img.append(img)
	aug_meas.append(meas)
	#Flip vertically to remove turn bias
	aug_img.append(cv2.flip(img, 1))
	aug_meas.append(meas*-1.0)
	

X_train = np.array(aug_img)
y_train = np.array(aug_meas)
print ('Augmented images: ', len(aug_img))

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers import Convolution2D, MaxPooling2D


model = Sequential()
model.add(Lambda(lambda x: x /255.0 - 0.5, input_shape=(160, 320, 3)))
#model.add(Cropping2d(cropping=((70,25),(0,0))))
model.add(Convolution2D(24,(5,5), activation="relu"))
model.add(Dropout(0.2))
model.add(MaxPooling2D())
model.add(Convolution2D(36, (5,5), activation="relu"))
model.add(Dropout(0.2))
model.add(MaxPooling2D())
model.add(Convolution2D(48, (5,5), activation="relu"))
model.add(Dropout(0.2))
model.add(MaxPooling2D())
model.add(Convolution2D(64, (3,3), activation="relu"))
model.add(Dropout(0.2))
model.add(MaxPooling2D())
model.add(Convolution2D(64, (3,3), activation="relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, batch_size=128, epochs=5, validation_split=0.2, shuffle=True)

model.save('model.h5')
#model.summary()
