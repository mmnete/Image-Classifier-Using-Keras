#Let us make all the necessary imports
import numpy as np #this is for matrix calculations and also preparing the input data matrix
#this is what we want to use from keras 
from keras.utils import np_utils 
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, RMSprop, adam

from random import randint
import random


#one more import that will help us read the images into arrays 
import cv2

#This import will help us find all the files in the directories where the data is saved ..
from os import listdir
from os.path import isfile, join

people_image_files = [f for f in listdir('dataset/prepared_people_images/')]
car_image_files = [f for f in listdir('dataset/prepared_car_images/')]


#Well, let's get the data now 
people_img = []
car_img = []
labels = []


#Now we need to prevent overfitting, and that is why we will do some augmentations 
#Augmentation is just simple putting some changes in the images 
#The one thing we will do is add blur to some of out images 
#We have around 408 samples and so what we will try to do is ...
#flip those images vertically and horizontally 
#This will help us get more images atleast 3 per input 


#get the car images and blur half of them
t = 0
for i in car_image_files:
    img = cv2.imread("dataset/prepared_car_images/"+i)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img1 = cv2.flip(img, 1)
    img2 = cv2.flip(img, 0)
    img = cv2.resize(img, (100,100))
    img1 = cv2.resize(img1, (100,100))
    img2 = cv2.resize(img2, (100,100))
    img3 = img
    car_img.append(img)
    car_img.append(img1)
    car_img.append(img2)
    car_img.append(img3)
    labels.append(np.array([1,0]))
    labels.append(np.array([1,0]))
    labels.append(np.array([1,0]))
    labels.append(np.array([1,0]))
    t += 1
t = 0

#get the people images and blur half of them 
for i in people_image_files:
    img = cv2.imread("dataset/prepared_people_images/"+i)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img1 = cv2.flip(img,1)
    img2 = cv2.flip(img, 0)
    img = cv2.resize(img ,(100,100))
    img1 = cv2.resize(img1, (100,100))
    img2 = cv2.resize(img2, (100,100))
    img3 = img
    people_img.append(img)
    people_img.append(img1)
    people_img.append(img2)
    people_img.append(img3)
    labels.append(np.array([0,1]))
    labels.append(np.array([0,1]))
    labels.append(np.array([0,1]))
    labels.append(np.array([0,1]))
    t += 1

input_images = []

for i in car_img:
    input_images.append(i)

for i in people_img:
    input_images.append(i)

#The reason why we change the input image list into a numpy array after the appending 
#is because numpy arrays do not append 

input_images = np.array(input_images)
labels = np.array(labels)


#Now we will reshape the inputs one more time
#We will prepare validation and training data
#the validaion data is the data that the model will not train on but will be used to 
#evaluate the model after training 
validation_input_images = [] #this is the validation input data 
validation_labels = [] #this is the validation output 
input_images1 = [] #this is the training input 
labels1 = [] #this is the training output 

for i in range(len(labels)):
    t = random.randint(1,100)
    if t%2 == 0:
        input_images1.append(input_images[i])
        labels1.append(labels[i])
    else:
        validation_input_images.append(input_images[i])
        validation_labels.append(labels[i])

    
    
input_images = np.expand_dims(input_images, axis=1)
input_images1 = np.expand_dims(input_images1, axis=1)
validation_input_images = np.expand_dims(validation_input_images, axis=1)
labels1 = np.array(labels1)
validation_labels = np.array(validation_labels)

#So now we have the data... let's prepare our model....
input_shape = input_images[0].shape

print(input_shape)

model = Sequential()
model.add(Convolution2D(32,3,3,border_mode='same',input_shape=input_shape))
model.add(Activation('relu'))
model.add(Convolution2D(32,3,3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.6))
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy', optimizer="rmsprop",metrics=["accuracy"])
model.fit(input_images1, labels1, batch_size = 50, nb_epoch=1, verbose = 1, validation_data=(validation_input_images,validation_labels))

#now we save the model so that we can use it later 
model.save('saved_model.h5')
model.save_weights('saved_model_weights.h5')




