So we want to build a machine learning model that will be able to classify images of cars from images of people.
The datasets used can be found here. 

http://www-old.emt.tugraz.at/~pinz/data/GRAZ_01/ - Images of People.
http://ai.stanford.edu/~jkrause/cars/car_dataset.html - Images of Cars.

Now to do this, you will need python downloaded, in order to know the keras version you need, please look that up on google. 
Just download python, numpy, cv2, and keras. After downloading all that you will be able to perform all these operations.
If you just want to test the model yourself, you can run the let_me_predict.py. Make sure you save it in the directory where the image you want 
to classify is, and it will prompt for the name of the image
Save all those images in two seperate directories.
We will then programmatically check those directories, get the filenames and add then to our input arrays 

Before you start please make the necessary imports
'''

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



'''

After that, try and run the program. If everything is okay, then you are ready to go. 
'''
  from os import listdir
from os.path import isfile, join

people_image_files = [f for f in listdir('dataset/prepared_people_images/')]
car_image_files = [f for f in listdir('dataset/prepared_car_images/')]


'''
At this point you should get all the image names in those two lists.

This code checks into the file directories prepared_car_images/ and prepared_people_images/ for all the files available 
and then puts the names of those files in a collection.

After that we need to prepare the data accordingly.
So we know that we have very few images. That means we need to augment the data so that we can get more data. A machine learns better with more data.
We will try to flip each image both vertically and horizontally, so that we know for each image we have 3 images 

We will do the code below for each input 
'''

t = 0
for i in car_image_files:
    img = cv2.imread("dataset/prepared_car_images/"+i)  ##Read the images file 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  ##change the image to black and white 
    img1 = cv2.flip(img, 1) ##flip the image vertically
    img2 = cv2.flip(img, 0) ##flip the image horizontally
    img = cv2.resize(img, (100,100)) ##resize all the images 
    img1 = cv2.resize(img1, (100,100))
    img2 = cv2.resize(img2, (100,100))
    img3 = img
    car_img.append(img) ##add all the different forms of that image as inputs 
    car_img.append(img1)
    car_img.append(img2)
    car_img.append(img3)
    labels.append(np.array([1,0])) ##add the respective labels of those images 
    labels.append(np.array([1,0]))
    labels.append(np.array([1,0]))
    labels.append(np.array([1,0]))
    t += 1
'''

Now it is time to create our model, we will be create our model and we will obviously do two things before we input our data
First of all we have to get the shape of the input data,
The code below does that, and also shows the model built
'''


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


'''

You obviously have to change the epoch and everything, but this model at most will have an accuracy of around 80%
As for what each layer means, let us break it down this way.
The first layer is obviously the input, the second one is the just an activation.
Now convolution layers usually take smaller portions of bigger images and summarize those as the inputs into the network. 
That is what the maxpooling layer does, it takes smaller images of 2 by 2. This is because processing the whole image is a bit more tough.
The flatten layer just changes the 2d input into 1d layers this is so that we can make the input into the output we want.
Our output is basically an array [1,0] for a car and [0,1] for a human being. 
After that we have the dropout layer. It basically randomly turns off 50% of the neurons, this is to reduce overfitting.
The rest is to help the function reach a minimum and there are several optimizers one can use for different scenerios and different 
data inputs.
So until this point if you run the program, and make sure you have the images in the appropriate directory, the model will compile 
with an accuracy of around 0.7 ish, better than a half. There are other techniques one can use to make the model better. 
At the end all I do is save the model and reload the model and the weights to make predictions on other images. 

