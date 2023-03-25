import cv2
import time
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

#Allowing the webcam to start by making the code sleep for 2 seconds
time.sleep(2)
bg = 0

#Capturing background for 60 frames
for i in range(60):
    ret, bg = cap.read()
#Flipping the background
bg = np.flip(bg, axis=1)

#Reading the captured frame until the camera is open
while (cap.isOpened()):
    ret, img = cap.read()
    if not ret:
        break
    #Flipping the image for consistency
    mountain = cv2.imread('mount everest.jpg')
    mountain1 = cv2.resize(mountain, (640, 480))

    hsv = cv2.cvtColor(mountain, cv2.COLOR_BGR2HSV)
    #Generating mask to detect red colour(values can be changed)
    lower_bound = np.array([100,100,100])
    upper_bound = np.array([255,255,255])
    mask_1 = cv2.inRange(hsv, lower_bound, upper_bound)

    #cv2.imshow("mask_1", mask_1)

    #Open and expand the image where there is mask 1 (color)
   # mask_1=cv2.morphologyEx(mask_1,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
   # mask_1=cv2.morphologyEx(mask_1,cv2.MORPH_DILATE,np.ones((3,3),np.uint8))

    #Selecting only the part that does not have mask one and saving in mask 2
   # mask_2=cv2.bitwise_not(mask_1)
    #cv2.imshow('mask_2',mask_2)

    #Keeping only the part of the images without the red color 
    #(or any other color you may choose)
    #res_1=cv2.bitwise_and(img,img,mask=mask_2)

    #Keeping only the part of the images with the red color
    res_2=cv2.bitwise_and(bg,bg,mask_1)

    #Generating the final output
    final_output = cv2.addWeighted(res_2,1)
    
    
    #Displaying the output to the user
    cv2.imshow("magic", final_output)
    cv2.waitKey(1)

cap.release()
out.release()
cv2.destroyAllWindows()