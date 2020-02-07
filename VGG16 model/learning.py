from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import os
IMG_SIZE=128
'''gen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1,
                         shear_range=0.15, zoom_range=0.1, channel_shift_range=10, horizontal_flip=True,featurewise_center=True)

'''


'''
path="D://Dk//face_rec//web_dataset//cropped_dataset//Arav//"
image=cv2.imread(path,0)
image=cv2.resize(image,(IMG_SIZE,IMG_SIZE))
image=np.reshape(image,(-1,IMG_SIZE,IMG_SIZE,1))



itr=gen.flow(image)
aug_image=[next(itr)[0].astype(np.uint8) for i in range(20)]

p=1
for file in  os.listdir(path):
    i=i.reshape(IMG_SIZE,IMG_SIZE,1)
    cv2.imwrite("D://Dk//face_rec//web_dataset//cropped_dataset//train//aug//Arav//"+str(p)+".jpg",i)
    p+=1
    #cv2.imshow("i",i)
    #cv2.waitKey(0)

'''







'''import numpy as np
a=np.ones((5,6),dtype=int)
print(a.shape,a.ndim)
a=a.reshape(2,5,3)

print(a)
print(a.shape,a.ndim)'''
'''from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# reshape to be [samples][width][height][channels]
print(X_train.shape,X_train.ndim)
X_train = X_train.reshape((X_train.shape[0], 28, 28, 1))
X_test = X_test.reshape((X_test.shape[0], 28, 28, 1))
print(X_train.shape,X_train.ndim)
print(X_train[0][0].shape,X_train[0][0].ndim)'''


'''import numpy as np
a=np.ones((5,5),dtype=int)
a=np.reshape(a,(1,5,5,1))
print(a)'''

'''rom mtcnn import MTCNN
import cv2



vc=cv2.VideoCapture("D:\intern\First Floor_Trim.mp4")
while 1:
    _,img=vc.read()
    cv2.imshow("fun",img)
    cv2.waitKey(1)
    img = cv2.cvtColor(cv2.imread("D://Dk//face_rec//web_dataset//cropped_dataset//test//62.jpg"), cv2.COLOR_BGR2RGB)
    detector = MTCNN()
    print(detector.detect_faces(img))
vc.release()
cv2.destroyAllWindows()'''

def blur(img):
    return (cv2.blur(img,(5,5)))

#correct code
'''
count=1
p=78
gen=ImageDataGenerator(preprocessing_function=blur)
#path="D://Dk//face_rec//web_dataset//cropped_dataset//Arav//"
path="D://Dk//face_rec//web_dataset//cropped_dataset//Ragul//"
for file in os.listdir(path):
    image=cv2.imread(os.path.join(path,file))
    image=cv2.resize(image,(128,128))
    image=image.reshape(-1,128,128,3)
    itr=gen.flow(image)
    aug_image=[next(itr)[0].astype(np.uint8) for i in range(count)]
    for i in aug_image:
        i=cv2.resize(i,(IMG_SIZE,IMG_SIZE))
        image=cv2.cvtColor(i,cv2.COLOR_RGB2GRAY)
        image=cv2.resize(image,(IMG_SIZE,IMG_SIZE))
        image=image.reshape(IMG_SIZE,IMG_SIZE,1)
        cv2.imwrite("D://Dk//face_rec//web_dataset//cropped_dataset//train//aug//Ragul//"+str(p)+".jpg",image)
        cv2.imwrite("D://friday//aug//Arav//"+str(p)+".jpg",i)        
        p+=1
'''

'''path="D://Dk//face_rec//web_dataset//cropped_dataset//Arav//1.jpg"
image=cv2.imread(path,0)
plt.plot(image[0],image[1])'''




from mtcnn.mtcnn import MTCNN
import face_recognition
import cv2

# initialise the detector class.
detector = MTCNN()

# load an image as an array
vc=cv2.VideoCapture("D://friday//day ragul.mp4")
while 1:
    _,frame=vc.read()
    image = face_recognition.load_image_file(frame)

# detect faces from input image.
    face_locations = detector.detect_faces(image)

# draw bounding box and five facial landmarks of detected face
    for face in zip(face_locations):
        (x, y, w, h) = face[0]['box']
        landmarks = face[0]['keypoints']
        cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
        for key, point in landmarks.items():
            cv2.circle(image, point, 2, (255, 0, 0), 6)

    cv2.imshow('image',image)
    cv2.waitKey(1)


cv2.destroyAllWindows()
