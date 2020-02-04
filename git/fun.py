import cv2 
import numpy as np 
template = cv2.imread('D:\overview\Intern\Screenshot (45).png',0)
#template=cv2.resize(template,(109,29))
#arr=np.ones((29,109))
#cv2.imshow("a",arr)
w, h = template.shape[::-1]
methods = ['cv2.TM_CCOEFF']
cap = cv2.VideoCapture("D:\overview\Intern\Gate(2).mp4") 
if (cap.isOpened()== False): 
    print("Error opening video file") 
while(cap.isOpened()): 
    ret, frame = cap.read() 
    img=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    if ret == True: 
	    #cv2.imshow('Frame', frame)
        res = cv2.matchTemplate(img,template,eval(methods[0]))
        #res1=cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        if eval(methods[0]) in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)

        img1=cv2.rectangle(img,top_left, bottom_right, 180, 2)
        cv2.imshow("image",img1)
    if cv2.waitKey(25) & 0xFF == ord('q'): 
            break
    if ret==False:
        break
cap.release()  
cv2.destroyAllWindows() 
