import cv2
import time
from faced import FaceDetector
from faced.utils import annotate_image
from for_raghu import check
face_detector = FaceDetector()

frame_path="D://front camera//front6//"
SAVE_PATH="D://front camera//MTCNN//front6//"
vc= cv2.VideoCapture("D://front camera//videos//front6.mp4")
k=1
value=0
value1=0
while True:
    c,img=vc.read()
    if c!=False: 
        #img=cv2.imread("D://Dk//face_rec//Dataset//test//1.png")
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Receives RGB numpy image (HxWxC) and
    # returns (x_center, y_center, width, height, prob) tuples. 
        bboxes = face_detector.predict(rgb_img,0.8)
        for x,y,w,h,p in bboxes:    
            if round(p,2)>0.90:
                x=int(x - w/2)
                y=int(y -h/2)
                img1=img[y:y+h,x:x+w]
                #path="D://Dk//face_rec//web_dataset//cropped_dataset//test//"+str(k)+".jpg"
                #seconds = time.time()
                #time.sleep(1)
                #print(k)
                #print("Seconds since epoch =", seconds)	
                #print(x,y)
                #cv2.waitKey(1)
                cv2.imwrite(frame_path+str(k)+".jpeg",img1)
                k+=1
        ann_img = annotate_image(img, bboxes)
    else:
        break
# Use this utils function to annotate the image.
        
    
# Show the image
    #ann_img=cv2.resize(ann_img,(ann_img.shape[0]//2,ann_img.shape[1]//2))
    cv2.imshow('image',ann_img)
    cv2.waitKey(1)
    



cv2.destroyAllWindows()
vc.release()


check(frame_path,SAVE_PATH)
'''
sharpen(path)
#delete the blured and faceless images
#
'''

