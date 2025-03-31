import cv2
import mediapipe as mp
import time

# initialize video capture
cap=cv2.VideoCapture("videos/face3.mp4")
pTime=0  # previous time for fps calculation

# initialize mediapipe face detection module
mpFace=mp.solutions.face_detection
mpDraw=mp.solutions.drawing_utils
faceDetector=mpFace.FaceDetection(0.75)  # confidence threshold

while True:
    success,img=cap.read()
    if not success:
        break  # exit loop if video ends

    imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)  # convert to rgb
    results=faceDetector.process(imgRGB)  # detect faces

    if results.detections:
        for id,detection in enumerate(results.detections):
            bboxC=detection.location_data.relative_bounding_box  # get bbox coordinates
            h,w,c=img.shape  # get image dimensions
            bbox=(int(bboxC.xmin*w),int(bboxC.ymin*h),
                  int(bboxC.width*w),int(bboxC.height*h))
            cv2.rectangle(img,bbox,(255,0,255),2)  # draw bounding box
            cv2.putText(img,f'{int(detection.score[0]*100)}%',
                        (bbox[0],bbox[1]-20),cv2.FONT_HERSHEY_PLAIN,
                        2,(255,0,255),2)  # display confidence

    # calculate and display fps
    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    cv2.putText(img,f'FPS: {int(fps)}',(20,70),cv2.FONT_HERSHEY_PLAIN,
                3,(0,255,0),2)
    cv2.namedWindow("Image",cv2.WINDOW_NORMAL)  # Make window resizable
    cv2.resizeWindow("Image",800,600)  # Set custom width and height
    cv2.imshow("Image",img)
    cv2.waitKey(1)
