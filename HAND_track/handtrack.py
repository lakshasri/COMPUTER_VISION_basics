# pip install mediapipe
# pip install opencv-python
import mediapipe as mp  # made by google
import cv2
import time  # check framerate

capture = cv2.VideoCapture(0)  # 0 in the sense 0th camera ie primary camera
# this code is used to run a webcam
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
pTime = 0
cTime = 0

while True:
    success, img = capture.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB conversion
    results = hands.process(imgRGB)
    # extract results #we might have multiple hands so for loop should be used in that case
    # print(results) #it will say none if no hand is detected else it will say landmark with cordinates
    # print(results.multi_hand_landmarks)
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                print(
                    id, lm
                )  # used to see whats happening with each hand ie unique id and corresponding landmark
                height, width, channel = img.shape
                cx, cy = (
                    int(lm.x * width),
                    int(lm.y * height),
                )  # cx and cy position of center
                print(
                    id, cx, cy
                )  # this is will for all 21 ids with cx cy pos used to print out landmarks for each pt.
                if id == 0:
                    cv2.circle(
                        img, (cx, cy), 15, (255, 0, 255), cv2.FILLED
                    )  # 255,0,255 is purple btw
            mpDraw.draw_landmarks(
                img, handLms, mpHands.HAND_CONNECTIONS
            )  # this will give hand with all 21 connections
    cTime = time.time()  # current time
    fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0  # frames per second = 1 / current time - prev time
    pTime = cTime  # prev time = current time
    cv2.putText(
        img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3
    )
    # str(int(fps)) covert to string because its time and int is used to round it off
    cv2.imshow("Image" ,img)  # changed imgRGB to img so the camera feed is displayed correctly
    cv2.waitKey(1)
