from keras.models import load_model
import cv2 
import numpy as np

model = load_model('saved_model.h5')
model.load_weights('saved_model_weights.h5', by_name=True)

def prepareImage(x):
    input_img = cv2.imread(x+".jpg")
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    input_img =  cv2.resize(input_img, (100,100))
    input1 = []
    input1.append(input_img)
    input1 = np.array(input1)
    input1 = np.expand_dims(input1, axis=1)
    return input1

on = True
while on:
    t = raw_input("Type in the name of the image you want me to predict, also please make sure the image is in this directory example imag1 without the extension. Make sure the image is jpg.")
    if not cv2.imread(t) is None:
        print("Please enter a valid image file in this directory. E.g image.jpg")
    else:
        input_img = prepareImage(t)
        print("I was only taught how to classify images of humans from images of cars, if you give me other wise, I wont be able to do it.")
        print("Well I think.....")
        print(model.predict(input_img, batch_size = 32, verbose = 0))
    c = raw_input("Do you want to continue y, for yes anythingelse for no?")
    if c == "y":
        on = True
    else:
        on = False
print("Bye.")

		


