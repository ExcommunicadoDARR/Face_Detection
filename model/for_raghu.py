from mtcnn.mtcnn import MTCNN
import cv2
import os

def check(frame_path,save_path):
    print("checking>>>>>>>>>>>>")
    files=os.listdir(frame_path)
    print("lenght",len(files))
    detector = MTCNN()
    l=0
    count=1
    for file in files:
        img=cv2.imread(os.path.join(frame_path,file))
        if img is not None:
            faces = detector.detect_faces(img)
            if len(faces)!=0 and round(faces[0]['confidence'],2)>0.85:
                x,y,w,h=faces[0]['box']
                if x>0 and y>0 and w>0 and h>0:
                    image=img[y:y+h,x:x+w]
                    image_256X256=cv2.resize(image,(256,256))
                    cv2.imwrite(save_path+str(count)+".jpeg",image_256X256)
                    #print(faces)
                    count+=1
                    print(count)
            
                
        '''
        if (img) is not None:
            faces = detector.detect_faces(img)
            if len(faces)==0:
                os.remove(os.path.join(frame_path,file))
                l+=1
            else:
               
                imag=cv2.GaussianBlur(img,(3,3),0)
                imag=cv2.Laplacian(imag,cv2.CV_64F).var()
                # image blur detection
                if imag>0:
                    print(file)
                    #os.remove(os.path.join(path,file))
                    x,y,w,h=faces[0]['box']
                    #cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),1)
                    image=img[y:y+h,x:x+w]
                    cv2.imwrite(os.path.join(save_path,file),image)
                    #os.remove(os.path.join(path,file))
                    #cv2.imshow("fun",image)
                    #cv2.waitKey(0)
        '''

    print("end>>>>>>>>>>>>>>>>>")

#sharpen image
'''
import cv2
import numpy as np
# Reading in the input image
image = cv2.imread('D://blur_model//dataset//train//Arav//1.jpeg')
# Create a matrix of ones of type int in the same size as the image
# then multiply it by a scaler of 75 
# This gives a matrix with same dimesions of our image with all values being 75
matrix = np.ones(image.shape, dtype = "uint8") * 75
# We use the matrix to add to our image
added = cv2.add(image, matrix)
cv2.imshow("Added", added)
cv2.waitKey(0)
# Likewise we can also subtract
subtracted = cv2.subtract(image, matrix)
cv2.imshow("Subtracted", subtracted)
# Wait & terminate
cv2.waitKey(0)
cv2.destroyAllWindows()'''



