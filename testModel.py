import cv2
import numpy as np
from trainModel import *

# Load Haar Cascade
cascade_path = 'cascades/haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)

# Start camera
capture = cv2.VideoCapture(0)

while True:
    ret, frame = capture.read()

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        cv2.putText(frame, "No Face Found", (50,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    else:
        for (x, y, w, h) in faces:
            # Draw rectangle around face
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

    # Show output
    cv2.imshow("Face Detection", frame)

    # Press ENTER to exit
    if cv2.waitKey(1) == 13:
        break

capture.release()
cv2.destroyAllWindows()