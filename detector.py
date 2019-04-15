import cv2
import numpy as np

faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0);
rec = cv2.face.LBPHFaceRecognizer_create();
rec.read("recognizer\\trainingData.yml")
id = 0
#font = cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_COMPLEX_SMALL,5,1,0,4)
fontface = cv2.FONT_HERSHEY_SIMPLEX
fontscale = 1
fontcolor = (255, 255, 255)

while(cam.isOpened()):
    ret, im = cam.read();
    if ret:
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = faceDetect.detectMultiScale(gray, 1.3, 5);
        for(x,y,w,h) in faces:
            cv2.rectangle(im,(x,y),(x+w, y+h),(0,255,0),2)
            id, conf = rec.predict(gray[y:y+h, x:x+w])
            if(id == 1):
                id = "Prataya"
            cv2.putText(im, str(id), (x,y+h), fontface, fontscale, fontcolor);
        cv2.imshow("face1", im);
        if(cv2.waitKey(1) == ord('q')):
            break;
cam.release()
cv2.destroyAllwindows()

        

    
