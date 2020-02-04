# OpenCV program to detect face in real time 
# import libraries of python OpenCV 
# where its functionality resides 
import cv2 
import os
# load the required trained XML classifiers 
# https://github.com/Itseez/opencv/blob/master/ 
# data/haarcascades/haarcascade_frontalface_default.xml 
# Trained XML classifiers describes some features of some 
# object we want to detect a cascade function is trained 
# from a lot of positive(faces) and negative(non-faces) 
# images. 
face_cascade = cv2.CascadeClassifier('D:\overview\Intern\haarcascade_frontalface_default.xml') 

# https://github.com/Itseez/opencv/blob/master 
# /data/haarcascades/haarcascade_eye.xml 
# Trained XML file for detecting eyes 
#eye_cascade = cv2.CascadeClassifier('D:\overview\Intern\haarcascade_eye.xml') 

# capture frames from a camera 
#cap = cv2.VideoCapture("D:\overview\Intern\First Floor Entrance 1(0).mp4") 
#D:\overview\Intern\First Floor Entrance 1(0).mp4
# loop runs if capturing has been initialized. 
total=0
while 1: 

	# reads frames from a camera 
	#_, img = cap.read() 
	img=cv2.imread("D:\overview\Intern\WIN_20200124_11_49_52_Pro (2).jpg")
	# convert to gray scale of each frames 
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

	# Detects faces of different sizes in the input image 
	faces = face_cascade.detectMultiScale(gray, 1.3, 2) 

	for (x,y,w,h) in faces: 
		# To draw a rectangle in a face 
		#cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),1) 
		roi_gray = gray[y:y+h, x:x+w] 
		roi_color = img[y:y+h, x:x+w] 
		imga=img[y-25:y+h+25,x-25:x+w+25]
		cv2.imshow("fuk",imga)
		#cv2.waitKey(0)
		print(x,y,x+w,y+h)
		# Detects eyes of different sizes in the input image 
		#eyes = eye_cascade.detectMultiScale(roi_gray) 

	'''	#To draw a rectangle in eyes 
		for (ex,ey,ew,eh) in eyes: 
			cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,127,255),2) 
    '''
	# Display an image in a window 
	#cv2.imshow('img',img) 

	# Wait for Esc key to stop 
	k = cv2.waitKey(30) & 0xff
	if k == ord("k"):
		p=os.path.sep.join(["{}.png".format(
			str(total).zfill(2))])
		cv2.imwrite(p,imga)	
		total +=1
	elif k == 27: 
		break

# Close the window 
#cap.release() 

# De-allocate any associated memory usage 
cv2.destroyAllWindows() 