import cv2
import mediapipe as mp
import time
#just to not be redudant of wasting too much storage i have just not added the 
# video files just use the video files i have used before for point estimation
cap=cv2.VideoCapture("../videos/1.mp4") # video input
pTime=0 

draw=mp.solutions.drawing_utils 
fmesh=mp.solutions.face_mesh 
faceMesh=fmesh.FaceMesh(max_num_faces=2) 
spec=draw.DrawingSpec(thickness=1,circle_radius=2) # thin lines, small circles

while True:
    success,img=cap.read()
    imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB) 
    results=faceMesh.process(imgRGB) 
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            draw.draw_landmarks(img,faceLms,fmesh.FACEMESH_TESSELATION,spec,spec) # draw face mesh
        for id,lm in enumerate(faceLms.landmark):
            ih,iw,ic=img.shape 
            x,y=int(lm.x*iw),int(lm.y*ih) 
            print(id,x,y) 
    cTime=time.time()
    fps=1/(cTime-pTime) 
    pTime=cTime
    cv2.putText(img,f'FPS: {int(fps)}',(20,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3) # display fps
    cv2.imshow("Image",img) # show output
    cv2.waitKey(1)