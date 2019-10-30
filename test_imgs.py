import os
import sys
import cv2
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, load_model
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i","--image",required = True,help = "Path to image")
args = vars(parser.parse_args())

img_width, img_height = 150, 150
model_path = './models/model.h5'
model_weights_path = './models/weights.h5'
model = load_model(model_path)
model.load_weights(model_weights_path)
image = cv2.imread(args["image"])

def predict(file):
  x = load_img(file, target_size=(img_width,img_height))
  x = img_to_array(x)
  x = np.expand_dims(x, axis=0)
  array = model.predict(x)
  result = array[0]
  answer = np.argmax(result)
  if answer == 0:
    return "M"
  elif answer == 1:
    return "V"
  elif answer == 2:
    return "W"
img = cv2.resize(image, (640,480), interpolation = cv2.INTER_AREA)
ans = predict(args["image"])
#print ans
cv2.putText(img, '%s' % (ans.upper()), (100,400), cv2.FONT_HERSHEY_SIMPLEX, 4, (0,255,0), 4)
cv2.imshow("img",img)
cv2.waitKey(0)
