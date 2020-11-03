import cv2
import time
import numpy as np

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640,480))



readFromWebcam=cv2.VideoCapture(0)

time.sleep(7)
count = 0
background = 0

for i in range(60):
    read, background=readFromWebcam.read()
background = np.flip(background, axis=1)

while(readFromWebcam.isOpened()):
    read, img=readFromWebcam.read()
    if not read:
        break
    count = count + 1
    img= np.flip(img, axis=1)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower = np.array([0,10,20])
    upper= np.array([220, 200, 255])
    mask1 = cv2.inRange(hsv, lower, upper)
    
    lower = np.array([10,110,120])
    upper= np.array([20, 0, 55])
    mask2 = cv2.inRange(hsv, lower, upper)

    mask1 = mask1 + mask2

    mask2 = cv2.bitwise_not(mask1)

    r1 = cv2.bitwise_and(img, img, mask=mask2)

    r2 = cv2.bitwise_and(background, background, mask=mask1)

    finalOutput = cv2.addWeighted(r1, 1, r2, 1, 0)
    out.write(finalOutput)
    cv2.imshow("magic", finalOutput)
    cv2.waitKey(1)

readFromWebcam.release()
out.release()
cv2.destroyAllWindows()
print("completed the program")
