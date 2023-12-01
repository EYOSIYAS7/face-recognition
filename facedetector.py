
import cv2


trained_face = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()


recognizer.read("faceRecog.yml")
# Capturing form the webcam

print("live or via image ")
user_input = input("Enter something: ")


def live_demo():
 webCam = cv2.VideoCapture(0)
 while True:

# reading a single frame at a time
    successfully_frame_read, frame = webCam.read()
    grayScaled = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    grayScaled = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# this will return a list of detected face coordinates and smile coordinates in the face 
    faceCordinates = trained_face.detectMultiScale(grayScaled)



    for (x,y,w,h) in faceCordinates:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        # extracting the face from the frame by slicing
        theFace = frame[y:y+h, x:x+w]
        grayFace = cv2.cvtColor(theFace, cv2.COLOR_BGR2GRAY)
 
        prediction , confidence = recognizer.predict(grayFace)

        # you can modify the strength by increasing the confidence value
        if confidence >= 85 and confidence <=95:
            print(prediction, confidence)
        if(prediction == 2):
           
            cv2.putText(frame, "eyosiyas", (x, y+h+60), fontScale=2, fontFace= cv2.FONT_HERSHEY_SCRIPT_SIMPLEX , color=(255,255,255), thickness =3 )
        elif(prediction == 1):
           
            cv2.putText(frame, " Walter white", (x, y+h+60), fontScale=2, fontFace= cv2.FONT_HERSHEY_SCRIPT_SIMPLEX , color=(255,255,255), thickness =3)
        elif(prediction == 0):
            
            cv2.putText(frame, "jessie Pinkman", (x, y+h+60), fontScale=2, fontFace= cv2.FONT_HERSHEY_SCRIPT_SIMPLEX , color=(255,255,255), thickness =3)
        elif(prediction == 3):
            
            cv2.putText(frame, "jessie pinkman", (x, y+h+60), fontScale=2, fontFace= cv2.FONT_HERSHEY_SCRIPT_SIMPLEX , color=(255,255,255), thickness =3)
    cv2.imshow("face detection app", frame)

    key = cv2.waitKey(1) 
# press esc or Q TO exit
    if key ==81 or key==113 or key == 27:
        break

# webCam.release() 
 cv2.destroyAllWindows()

def imge_demo(url):
 cap = cv2.imread(url)

 while True:

# reading a single frame at a time
    # successfully_frame_read, frame = webCam.read()
    grayScaled = cv2.cvtColor(cap, cv2.COLOR_BGR2GRAY)
    grayScaled = cv2.cvtColor(cap, cv2.COLOR_BGR2GRAY)
# this will return a list of detected face coordinates and smile coordinates in the face 
    faceCordinates = trained_face.detectMultiScale(grayScaled)



    for (x,y,w,h) in faceCordinates:
        cv2.rectangle(cap, (x,y), (x+w,y+h), (0,255,0), 2)
        # extracting the face from the frame by slicing
        theFace = cap[y:y+h, x:x+w]
        grayFace = cv2.cvtColor(theFace, cv2.COLOR_BGR2GRAY)
 
        prediction , confidence = recognizer.predict(grayFace)
        print(prediction, confidence)
        if confidence >= 45 and confidence <=85:
            print(prediction, confidence)
        if(prediction == 2):
            cv2.putText(cap, "eyosiyas", (x, y+h+60), fontScale=2, fontFace= cv2.FONT_HERSHEY_SCRIPT_SIMPLEX , color=(255,255,255), thickness =3 )
        elif(prediction == 1):
            cv2.putText(cap, " Walter white", (x, y+h+60), fontScale=2, fontFace= cv2.FONT_HERSHEY_SCRIPT_SIMPLEX , color=(255,255,255), thickness =3)
        elif(prediction == 3):
            cv2.putText(cap, "Jessie pinkman", (x, y+h+60), fontScale=2, fontFace= cv2.FONT_HERSHEY_SCRIPT_SIMPLEX , color=(255,255,255), thickness =3)     
        elif(prediction == 0):
            
            cv2.putText(cap, "Jessie pinkman", (x, y+h+60), fontScale=2, fontFace= cv2.FONT_HERSHEY_SCRIPT_SIMPLEX , color=(255,255,255), thickness =3)
            cv2.putText(cap, " ", (x, y+h+60), fontScale=2, fontFace= cv2.FONT_HERSHEY_SCRIPT_SIMPLEX , color=(255,255,255), thickness =3)
    cv2.imshow("face detection app", cap)

    key = cv2.waitKey(1) 
# press esc or Q TO exit
    if key ==81 or key==113 or key == 27:
        break

# webCam.release() 
 cv2.destroyAllWindows()


if user_input == "live":
    live_demo()
# 
elif user_input == "image":
  url = input("input image url:" )
  imge_demo(url)

print("code completed ")
