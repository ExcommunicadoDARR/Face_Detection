import pickle
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers import MaxPooling2D,BatchNormalization
from keras.layers.core import Dense,Flatten,Activation,Dropout
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
class_lables=["Arav","Ragul"]

X = pickle.load(open("D://front camera aug//model//X.pickle", "rb"))
Y = pickle.load(open("D://front camera aug//model//Y3.pickle", "rb"))

X.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
print(X.shape,X.ndim)


'''model=keras.Sequential([
keras.layers.Flatten(input_shape=(100,100,1)),
keras.layers.Dense(128,activation=tf.nn.relu),
keras.layers.Dense(2,activation=tf.nn.softmax)                      
])'''


#not working
'''model=Sequential([
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
    Dense(4,activation="softmax")
]) 




model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history=model.fit(X,Y,epochs=5,shuffle=True,validation_split=0.2)

'''


#keras model
'''
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(IMG_SIZE,IMG_SIZE,1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(5))
model.add(Activation('sigmoid'))

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

'''

model = Sequential()
model.add(Conv2D(16, (3,3), activation='relu',input_shape=(IMG_SIZE,IMG_SIZE,1)))
model.add(BatchNormalization())
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.3))

model.add(Conv2D(32, (3,3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.3))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.4))

model.add(Conv2D(128, (3,3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))


model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))


model.add(Dense(2, activation='sigmoid'))


model.compile(optimizer='adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])

history=model.fit(X,Y,epochs=5,shuffle=True,validation_split=0.1)


plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


 
'''print(history.history)
#plot(history)
print(model.summary())'''

#plot(history,5)


model_json = model.to_json()
with open("D://front camera aug//model//model.json", "w") as json_file :
	json_file.write(model_json)

model.save_weights("D://front camera aug//model//model.h5")
print("Saved model to disk")

model.save('D://front camera aug//model//CNN.model')