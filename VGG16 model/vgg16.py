import pickle
import keras
from keras import optimizers
from keras.layers.convolutional import Conv2D
from keras.layers import MaxPooling2D
from keras.layers.core import Dense,Flatten
import numpy as np
import matplotlib.pyplot as plt
import cv2
from keras.models import model_from_json
from keras.models import load_model
from plot import plot

IMG_SIZE=224
CATEGORIES = ["Arav","Ragul","Raghu","dhana","Non"]
X = pickle.load(open("X_color.pickle", "rb"))
Y = pickle.load(open("Y_color.pickle", "rb"))

'''gg16= keras.applications.vgg16.VGG16()
print(type(vgg16))
model=keras.Sequential() 
for layer in vgg16.layers:
    model.add(layer)
#model.layers.pop()


for layer in model.layers:
    layer.trainable=False
model.add(Dense(5,activation="softmax"))

sgd = optimizers.SGD(lr=0.0001, clipvalue=0.5)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
his=model.fit(X,Y,shuffle=True,validation_split=0.2,epochs=5)
print(model.summary()) 
plot(his,5)'''

model=keras.Sequential([
    Conv2D(16,(3,3),activation="relu",input_shape=(IMG_SIZE ,IMG_SIZE,3)),
    Conv2D(16,(3,3),activation="relu"),
    MaxPooling2D(pool_size =(2, 2),strides=(2,2)),
    Conv2D(32,(3,3),activation="relu"),
    Conv2D(32,(3,3),activation="relu"),
    MaxPooling2D(pool_size =(2, 2),strides=(2,2)),
    Conv2D(64,(3,3),activation="relu"),
    Conv2D(64,(3,3),activation="relu"),
    Conv2D(64,(3,3),activation="relu"),
    MaxPooling2D(pool_size =(2, 2),strides=(2,2)),
    Flatten(),
    Dense(5,activation="softmax")
]) 

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history=model.fit(X,Y,epochs=5,shuffle=True,validation_split=0.2)
plot(history,5)

#print(model.summary())

model_json = model.to_json()
with open("model1.json", "w") as json_file :
	json_file.write(model_json)

model.save_weights("model1.h5")
print("Saved model to disk")

model.save('CNN1.model')


'''import cv2
imag=cv2.imread("D://Vgg16//dataset//Dhana//103.jpeg")

imag=cv2.GaussianBlur(imag,(3,3),0)
imag=cv2.Laplacian(imag,cv2.CV_64F).var()
print(imag)
imag=imag/imag.max()
cv2.imshow("fun",imag)
cv2.waitKey(0)
cv2.destroyAllWindows()'''