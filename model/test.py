import cv2
import tensorflow as tf
import numpy as np
import os

def prepare(file):
    IMG_SIZE = 256
    img_array = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    #img_array = cv2.imread(file)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    #cv2.imshow("dawd",new_array)
    #cv2.waitKey(0)
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

model = tf.keras.models.load_model("D://front camera aug//model//CNN.model")
CATEGORIES = ["Arav","Ragul"]
#old model
'''
print(model.summary())
A_file=list()
R_file=list()
Non_file=list()
A=0
R=0
Non=0
Unknown=0
Confi_less=list()
path="D://front camera//dataset//test//"

l=len(os.listdir(path))

Arav_path="D://blur_model//output//Arav//"
#Non_path="D://path_approach//output//Non//"
Rahul_path="D://blur_model//output//Rahul//"
Unknown_path="D://blur_model//output//Unknown//"



for file in os.listdir(path):
    image_path=os.path.join(path,file)
    image=cv2.imread(image_path)
    i=prepare(os.path.join(path,file))
    prediction=model.predict(i)
    print(prediction,end="")
    if  np.argmax(prediction[0])>0.25:
        Unknown+=1
        cv2.imwrite(Unknown_path+file,image)
    else:
        if np.argmax(prediction[0])==1:
            R+=1
            R_file.append(file)
            cv2.imwrite(Rahul_path+file,image)
        elif np.argmax(prediction[0])==0:
            A+=1
            cv2.imwrite(Arav_path+file,image)
            A_file.append(file)
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

#print(Confi_less)'''

#new model
dic={}
for i in CATEGORIES:
    dic[i]=0
print(dic)
save_path="D://front camera aug//dataset//output//"
path="D://front camera aug//dataset//test//"
#path="D://front camera//dataset//train//ragul//"
for file in os.listdir(path):
    i=prepare(os.path.join(path,file))
    prediction=(model.predict(i)).tolist()
    value=CATEGORIES[prediction[0].index(max(prediction[0]))]
    print(prediction[0],file,max(prediction[0]))
    if round(max(prediction[0]),3)>0.50:
        cv2.imwrite(save_path+value+"//"+file,cv2.imread(os.path.join(path,file)))
    else:
        cv2.imwrite(save_path+"new//"+file,cv2.imread(os.path.join(path,file)))
    dic[value]=dic[value]+1
print(dic)

