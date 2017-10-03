#CD Vijay Udacity, CarND P3

#p34nvmod2.py: This is the Nvidia pipeline for P3, becoz of memory limitations in AWS, I added MaxPooling2D() layers.
#p34nvmod3.py: Added a few dropout layers
#p34nvmod4.py: Removed dropout layers
#p34nvmod5.py: Added dropout after flattening, Monitored accuracy etc. 
#p34nvmod52.py: Rmv.ed dropout after flattening, Monitored accuracy etc.

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np  
import cv2
import csv

def PlotAccuracyCurves():

	# summarize history for accuracy
	plt.plot(history.history['acc'])
	plt.plot(history.history['val_acc'])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()
	# summarize history for loss
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()



def compare_images(left_image, right_image):    
    print(image.shape)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    f.tight_layout()
    ax1.imshow(left_image)
    ax1.set_title('Shape '+ str(left_image.shape),
                  fontsize=25)
    ax2.imshow(np.uint8(right_image))
    ax2.set_title('Shape '+ str(right_image.shape)
                  , fontsize=25)
    plt.show()


lines = []

with open('driving_log.csv') as csvfile:
	reader = csv.reader (csvfile)
	for line in reader:
		lines.append(line)

images = []
measurements = []
for line in lines[1:]:
    source_path = line[0] 
	#drive.py sends RGB images, while OpenCv reads BGR, so using matplotlib to read
    rgbimage = mpimg.imread(source_path)
    #plt.imshow(rgbimage)
    #plt.show()
	
    images.append(rgbimage)
	
    measurement = float(line[3])
    measurements.append(measurement)
	
#print (len(images))	
#crop img to rid it of the useless pixels	top 70 pixels and btm 20 pixels, retain entire width
# Crop from x, y, w, h -> 0:0 origin, 320, 400
# NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]
print ('\n More augmentation in process...')
aug_img, aug_meas = [], []
for img, meas, in zip (images, measurements):
	#print (img.shape, type(image))
	#First shrink the size of image by half
	#img = cv2.resize(img, (32,32), interpolation = cv2.INTER_CUBIC)
	#Crop first before appending
	#img = img[70:140, 0:320] #Rmv top 70px, btm 20pix, w=320, so y goes from 70:140 px, X goes from 0:320, top left vertext = (x1=0,y1=70), ight btm vertext = (x2:320, y2:140), so for the numpy array: it is img[y1:y2, x1:x2]
	aug_img.append(img)
	aug_meas.append(meas)
	#Flip vertically to remove turn bias
	aug_img.append(cv2.flip(img, 1))
	aug_meas.append(meas*-1.0)
	
X_train = np.array(aug_img)
y_train = np.array(aug_meas)
print ('Augmented images: ', len(aug_img))

X_train = X_train.astype('float32')
y_train = y_train.astype('float32')

from keras.models import Sequential
from keras.layers.convolutional import Cropping2D 
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers import Convolution2D, MaxPooling2D, Cropping2D
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator

#TRAINING PIPELINE

model = Sequential()

model.add(Lambda(lambda x: x /255.0 - 0.5, input_shape=(160, 320, 3)))
#model.add(Lambda(lambda x: (x - 128.0)/ 128., input_shape=(160, 320, 3)))
#If cropping2d contains a tuple of 2 tuples of 2 ints: interpreted as ((top_crop, bottom_crop), (left_crop, right_crop))
#model.add(Cropping2d(cropping=((70,25),(0,0))))  this gives a strange error, 70 value doe snot sit well here
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160, 320, 3)))
#model.add(Cropping2D(cropping=((30,20), (0,0)), input_shape=(160, 320, 3)))
#cropping_output = K.function([model.layers[0].input], [model.layers[0].output])
#cropped_image = cropping_output([image[None,...]])[0]
#compare_images(image, cropped_image.reshape(cropped_image.shape[1:]))

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

model.compile(loss='mse', optimizer='adam',  metrics=['accuracy'])
history = model.fit(X_train, y_train, batch_size=64, epochs=5, validation_split=0.2, shuffle=True)
#PlotAccuracyCurves()

import sys
fname = sys.argv[0]
model.save((fname.replace(".py", ".h5")))
#model.summary()