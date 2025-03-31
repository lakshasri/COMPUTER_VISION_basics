import mediapipe as mp  # made by google
import cv2
import time  # check framerate

class handdetect():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode  # whether to treat input images as a video stream
        self.maxHands = maxHands  # maximum number of hands to detect
        self.detectionCon = detectionCon  # confidence threshold for detecting hands
        self.trackCon = trackCon  # confidence threshold for tracking hands
        self.mpHands = mp.solutions.hands  # mediapipe hands module
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode, 
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon, 
            min_tracking_confidence=self.trackCon
) #in this func we are using mediapipe inbuild func to detect hands so the colour is red when hand is tracked (just in case there is a confusion)
  
        self.mpDraw = mp.solutions.drawing_utils  

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert BGR to RGB for mediapipe processing
        self.results = self.hands.process(imgRGB)  # process the image to detect hands
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)  # draw hand landmarks
        return img  

    def findPosition(self, img, handNo=0, draw=True):
        lmList = []  
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]  
            for id, lm in enumerate(myHand.landmark):
                height, width, channel = img.shape  
                cx, cy = int(lm.x * width), int(lm.y * height)  
                lmList.append([id, cx, cy]) 
                if draw:
                    cv2.circle(img, (cx, cy), 5, (155, 44, 55), cv2.FILLED)  #here the colour of the hand will be some blue cause we chose 155, 44, 55 we are manually drawing the hand here 
        return lmList  

capture = cv2.VideoCapture(0)  # 0 in the sense 0th camera ie primary camera
# this code is used to run a webcam
pTime = 0  # previous time for calculating FPS
cTime = 0  # current time for calculating FPS

detector = handdetect()  # create an instance of the hand detector
while True:
    success, img = capture.read()  # read a frame from the webcam
    img = detector.findHands(img)  # detect hands and draw landmarks on the frame
    lmList = detector.findPosition(img)  # get the positions of hand landmarks
    if len(lmList) != 0:
        print(lmList[4])  # print coordinates of landmark 4 (tip of thumb)
    cTime = time.time()  # get the current time
    fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0  # calculate FPS
    pTime = cTime  # update previous time
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)  # display FPS on screen
    # str(int(fps)) converts FPS value to string and int rounds it off
    cv2.imshow("Image", img)  # changed imgRGB to img so the camera feed is displayed correctly
    cv2.waitKey(1)  # wait for a short time before capturing the next frame