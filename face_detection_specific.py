import cv2
import sys

#cascPath = sys.argv[0]
#faceCascade = cv2.CascadeClassifier(cascPath)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')
nose_cascade = cv2.CascadeClassifier('haarcascade_mcs_nose.xml')
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    faces = face_cascade.detectMultiScale(
      gray,
      scaleFactor=1.1,
      minNeighbors=5,
      minSize=(30, 30),
      flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )
    eyes = eye_cascade.detectMultiScale(gray,1.1,4)
    smile = smile_cascade.detectMultiScale(gray,1.1,100)
    nose = nose_cascade.detectMultiScale(gray,1.1,100)

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(frame, (ex, ey), (ex+ew, ey+eh), (255,0, 0), 2)
        for(sx,sy,sw,sh) in smile:
            cv2.rectangle(frame, (sx, sy), (sx+sw, sy+sh), (0,0,255), 2)
        for(nx,ny,nw,nh) in nose:
            cv2.rectangle(frame, (nx, ny), (nx+nw, ny+nh), (255,255,255), 2)



    # Display the resulting frame
    cv2.imshow('Video', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()