import face_recognition
import cv2
import os
import time
vid_path="sample1.mp4"

model = 'hog'
scale = 1
cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_1.avi', fourcc, 20.0, (256,256))
start = time.time();

def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

while True:
   
    ret,img = cap.read()
    if ret == True:
        img= cv2.resize(img,(620,480))
        img=increase_brightness(img)
        face_locations = face_recognition.face_locations(img, number_of_times_to_upsample=scale, model=model)
        for (top, right, bottom, left) in face_locations:
            ROI = img[top:bottom, left:right]
            blur = cv2.GaussianBlur(ROI, (51,51), 0)
            img[top:bottom, left:right]=blur
        out.write(img)
        cv2.imshow('IMG', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
end=time.time()
print(end-start)
cap.release()
out.release()
cv2.destroyAllWindows()
