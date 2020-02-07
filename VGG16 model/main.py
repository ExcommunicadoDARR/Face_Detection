import pickle
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers import MaxPooling2D
from keras.layers.core import Dense,Flatten
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import cv2
from keras.models import model_from_json
from keras.models import load_model
from plot import plot

'''vg16=keras.applications.vgg16.VGG16()
print(vg16.summary())
'''




IMG_SIZE=256
CATEGORIES = ["Arav","Ragul","Raghu","dhana","Non"]
X = pickle.load(open("X.pickle", "rb"))
Y = pickle.load(open("Y.pickle", "rb"))

X.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
print(X.shape,X.ndim)


'''model=keras.Sequential([
keras.layers.Flatten(input_shape=(100,100,1)),
keras.layers.Dense(128,activation=tf.nn.relu),
keras.layers.Dense(2,activation=tf.nn.softmax)                      
])'''
model=Sequential([
    Conv2D(16,(3,3),activation="relu",input_shape=( IMG_SIZE ,IMG_SIZE,1)),
    MaxPooling2D(pool_size =(3, 3)),
    Conv2D(32,(3,3),activation="relu"),
    MaxPooling2D(pool_size =(1, 1)),
    Conv2D(64,(3,3),activation="relu"),
    MaxPooling2D(pool_size =(3, 3)),
    Conv2D(128,(3,3),activation="relu"),
    MaxPooling2D(pool_size =(3, 3)),
    Conv2D(256,(3,3),activation="relu"),
    MaxPooling2D(pool_size =(3, 3)),
    Flatten(),
    Dense(5,activation="softmax")
]) 




model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history=model.fit(X,Y,epochs=10,shuffle=True,validation_split=0.2)


print(history.history)
#plot(history)
print(model.summary())

#plot(history,10)

model_json = model.to_json()
with open("model1.json", "w") as json_file :
	json_file.write(model_json)

model.save_weights("model1.h5")
print("Saved model to disk")

model.save('CNN1.model')