import cv2 
import mediapipe as mp 
import time 
mppose=mp.solutions.pose
pose=mppose.Pose()

mpdraw=mp.solutions.drawing_utils
cap=cv2.VideoCapture("videos/2.mp4") #depending on where you want to do post estimation, change the part inside the brackets 
ptime=0
while True:
    success,img=cap.read()
    imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results=pose.process(imgRGB)
    #print(results.pose_landmarks)
    if results.pose_landmarks:
        draw_spec = mpdraw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3)  # green lines, thicker
        mpdraw.draw_landmarks(img, results.pose_landmarks, mppose.POSE_CONNECTIONS, draw_spec, draw_spec)

        mpdraw.draw_landmarks(img,results.pose_landmarks,mppose.POSE_CONNECTIONS) #pose landmarks are the red dots and the connects is made using the next parameter 
        for id,lm in enumerate(results.pose_landmarks.landmark):
            h,w,c=img.shape #height width colour 
            print(id,lm)
            cx,cy=int(lm.x*w),int(lm.y*h)
            #cv2.circle(img,(cx,cy),10,(255,0,0),cv2.FILLED) #to check if it is working or overlapy is proper 
    ctime =time.time()
    fps=1/(ctime-ptime)
    ptime=ctime
    cv2.putText(img,str(int(fps)),(70,50),cv2.FONT_HERSHEY_DUPLEX,3,(124,222,56),3)
    img=cv2.resize(img,(800,600))
    cv2.imshow("Image",img)
    cv2.waitKey(10)