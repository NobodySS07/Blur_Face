import cv2
import time
vid_path="2022051011415420220510121154.mp4"
video_capture = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_haar.avi', fourcc, 20.0, (128,128))

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

# open
while True:
    try:
        ret, frame = video_capture.read()
        if ret ==True:
            frame = cv2.resize(frame, (640, 480))
            frame=increase_brightness(frame)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (top, right, bottom, left) in faces:
                face = frame[right:right+left, top:top+bottom]
                face = cv2.GaussianBlur(face,(51, 51), 30)
                frame[right:right+face.shape[0], top:top+face.shape[1]] = face
        
            out.write(frame)
            cv2.imshow('Face Blur', frame)
        else:
            break


    except Exception as e:
        print(f'exc: {e}')
        pass

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
out.release()
cv2.destroyAllWindows()
