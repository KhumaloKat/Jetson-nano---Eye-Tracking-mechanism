import cv2
import ObjectDetectionModule as odm
import SerialModule as sm
import numpy as np

# Camera setup
frameWidth = 640
frameHeight = 480
flip =  0

camSet='nvarguscamerasrc sensor_id=1 !  video/x-raw(memory:NVMM), width=3264,height=2464, format=NV12, framerate=21/1 ! nvvidconv flip-method='+str(flip)+'! video/x-raw, width='+str(frameWidth)+', height='+str(frameHeight)+',format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'

cap = cv2.VideoCapture(0)

# Serial connection setup
ser = sm.initConnection('/dev/ttyACM0', 9600)
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Init error variables
perrorLR, perrorUD = 0, 0

# Init Servo position
sm.sendData(ser, [90, 90], 3)


def trackObject(cx, cy, w, h):
    global perrorLR, perrorUD
    kLR = [0.7, 0.2]
    kUD = [0.7, 0.2]
    
    if cx != -1:
        # Left and Right
        errorLR = w // 2 - cx
        posX = kLR[0] * errorLR + kLR[1] * (errorLR - perrorLR)
        posX = int(np.interp(posX, [-w // 2, w // 2], [20, 160]))
        perrorLR = errorLR
        
        # Up and Down
        errorUD = h // 2 - cy
        posY = kUD[0] * errorUD + kUD[1] * (errorUD - perrorUD)
        posY = int(np.interp(posY, [-h // 2, h // 2], [160, 20]))
        perrorUD = errorUD
        
        sm.sendData(ser, [posX, posY], 3)

def findCenter(imgObjects, objects):
    cx, cy = -1, -1
    if len(objects) != 0:
        
        x, y, w, h = objects[0]
        cx = x + w // 2
        cy = y + h // 2

        cv2.circle(imgObjects, (cx, cy), 2, (0, 255, 0), cv2.FILLED)
        ih, iw, _ = imgObjects.shape
        cv2.line(imgObjects, (iw // 2, cy), (cx, cy), (0, 255, 0), 1)
        cv2.line(imgObjects, (cx, ih // 2), (cx, cy), (0, 255, 0), 1)

    return cx, cy, imgObjects

while True:
    _, img = cap.read()

    if not _ :
        break

    img = cv2.resize(img, (0,0), None, 0.3,0.3)
    
    imgObjects, objects = odm.findObjects(img, faceCascade, 1.1, 12)
    cx, cy, imgObjects = findCenter(imgObjects, objects)
    
    h, w, _ = imgObjects.shape
    cv2.line(imgObjects, (w // 2, 0), (w // 2, h), (255, 0, 255), 1)
    cv2.line(imgObjects, (0, h // 2), (w, h // 2), (255, 0, 255), 1)
    
    trackObject(cx, cy, w, h)
    img = cv2.resize(img, (0, 0), None, 4,4)
    cv2.imshow("Image", imgObjects)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        sm.sendData(ser, [45, 45], 3)
        break

cap.release()
cv2.destroyAllWindows()
