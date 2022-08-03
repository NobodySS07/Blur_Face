import cv2
import numpy as np
import matplotlib.pyplot as plt
lower = np.array([0, 35, 75], dtype = "uint8")
upper = np.array([15, 255, 255], dtype = "uint8")
vid = cv2.VideoCapture('2022051011415420220510121154.mp4')

def skin_detect(frame):
    
    converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    skinMask = cv2.inRange(converted, lower, upper)
# apply a series of erosions and dilations to the mask
# using an elliptical kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skinMask = cv2.erode(skinMask, kernel, iterations = 2)
    skinMask = cv2.dilate(skinMask, kernel, iterations = 2)
# blur the mask to help remove noise, then apply the
# mask to the frame
    skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
        
    skin = cv2.bitwise_and(frame, frame, mask = skinMask)
    
    return skin


while True:
    ret,frame = vid.read()
    if ret == False : break
    #frame = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    color = np.array([40,58,80], dtype='uint8')
    s = skin_detect(frame)
    cv2.imshow("s",s)
    #frame = np.where(s[...], color, frame)
    # blurred_img = cv2.GaussianBlur(frame, (51, 51), 7)
    frame = np.where(s[...], color, frame)
    
    #print(frame.shape)
    cv2.imshow("Output", frame)
    
    
    if cv2.waitKey(25) & 0xFF == ord('q'): 
            break
cv2.imshow("blur",frame)        
vid.release()
cv2.destroyAllWindows()

