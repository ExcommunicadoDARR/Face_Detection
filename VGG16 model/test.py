import cv2
import tensorflow as tf
import numpy as np
import os

def prepare(file):
    IMG_SIZE = 224
    img_array = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    #img_array = cv2.imread(file)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    print(new_array.shape)
    #cv2.imshow("dawd",new_array)
    #cv2.waitKey(0)
    return new_array.reshape( -1,IMG_SIZE, IMG_SIZE, 1)

model = tf.keras.models.load_model("CNN.model")
print(model.summary())
A_file=list()
R_file=list()
Non_file=list()
A=0
R=0
Raghu=0
D=0
Non=0
Unknown=0
Confi_less=list()
path="D://VGG16//frames_1//"

l=len(os.listdir(path))

Arav_path="D://VGG16//output//Arav//"
Non_path="D://VGG16//output//Non//"
Rahul_path="D://VGG16//output//Rahul//"
Raghu_path="D://VGG16//output//Raghu//"
Dhana_path="D://VGG16//output//Dhana//"
Unknown_path="D://VGG16//output//Unknown//"



for file in os.listdir(path):
    image_path=os.path.join(path,file)
    image=cv2.imread(image_path)
    i=prepare(os.path.join(path,file))
    prediction=model.predict(i)
    print(prediction,end="")
    '''if  np.argmax(prediction[0])>0.25:
        Unknown+=1
        cv2.imwrite(Unknown_path+file,image)'''
    if np.argmax(prediction[0])==1:
        R+=1
        R_file.append(file)
        cv2.imwrite(Rahul_path+file,image)
    elif np.argmax(prediction[0])==0:
        A+=1
        cv2.imwrite(Arav_path+file,image)
        A_file.append(file)
    elif np.argmax(prediction[0])==2:
        Raghu+=1
        cv2.imwrite(Raghu_path+file,image)
    elif np.argmax(prediction[0])==3:
        D+=1
        cv2.imwrite(Dhana_path+file,image)
    else:
        Non+=1
        cv2.imwrite(Non_path+file,image)
        Non_file.append(file)
    print(np.argmax(prediction[0]))


print("aravind:",(A/l*100))
#print(A_file)
print("Rahul:",R/l*100)
#print(R_file)
print("Non:",Non/l *100)
#print(Non_file)
print("confidentless",Unknown/l *100)
print("Raghu",Raghu/l *100)
print("dhana",D/l *100)

#print(Confi_less)