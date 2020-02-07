
import numpy as np
import os
from matplotlib import pyplot as plt
import cv2
import random
import pickle


file_list = []
class_list = []

DATADIR = "D://vgg16//dataset//"

# All the categories you want your neural network to detect
CATEGORIES = ["Arav","Ragul","Raghu","dhana","Non"]

# The size of the images that your neural network will use
IMG_SIZE = 224

# Checking or all images in the data folder
for category in CATEGORIES :
	path = os.path.join(DATADIR, category)
	for img in os.listdir(path):
		#img_array = cv2.imread(os.path.join(path, img))
		img_array = cv2.imread(os.path.join(path, img))#, cv2.IMREAD_GRAYSCALE)

training_data = []

def create_training_data():
	for category in CATEGORIES :
		path = os.path.join(DATADIR, category)
		class_num = CATEGORIES.index(category)
		for img in os.listdir(path):
			try :
				img_array = cv2.imread(os.path.join(path, img))#,cv2.IMREAD_GRAYSCALE)
				#img_array = cv2.imread(os.path.join(path, img))
				new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
				training_data.append([new_array, class_num])
			except Exception as e:
				pass

create_training_data()

random.shuffle(training_data)

X = [] #features
y = [] #labels

for features, label in training_data:
	X.append(features)
	y.append(label)
X = np.array(X)


X = np.array(X).reshape(-1,IMG_SIZE, IMG_SIZE, 3)

print(X.shape)

# Creating the files containing all the information about your model
pickle_out = open("X_color.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("Y_color.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()

#pickle_in = open("X.pickle", "rb")
#X = pickle.load(pickle_in)



