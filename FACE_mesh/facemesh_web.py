import cv2
import mediapipe as mp
import time

cap=cv2.VideoCapture(0)  # open webcam, because how else will we see your beautiful face?
pTime=0

draw=mp.solutions.drawing_utils
fmesh=mp.solutions.face_mesh
faceMesh=fmesh.FaceMesh(max_num_faces=2)
spec=draw.DrawingSpec(thickness=1,circle_radius=2)  # making the landmarks visible but not overkill

while True:
    success,img=cap.read()
    if not success:
        break  # if the webcam feed fails, we exit before causing chaos
    
    imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)  # converting because Mediapipe loves RGB, unlike OpenCV
    results=faceMesh.process(imgRGB)
    
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            draw.draw_landmarks(img,faceLms,fmesh.FACEMESH_TESSELATION,spec,spec)  # connecting the dots, literally
        
        for id,lm in enumerate(faceLms.landmark):
            ih,iw,ic=img.shape
            x,y=int(lm.x*iw),int(lm.y*ih)
            print(id,x,y)  # printing landmark positions, because why not?

    cTime=time.time()
    fps=1/(cTime-pTime)  # calculating FPS so we know if our code is crawling or flying
    pTime=cTime
    
    cv2.putText(img,f'FPS: {int(fps)}',(20,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)  # displaying FPS, because speed matters!
    cv2.imshow("Webcam Face Mesh",img)
    if cv2.waitKey(1)&0xFF==ord('q'):  # pressing 'q' to exit, because we all need an exit strategy
        break

cap.release()
cv2.destroyAllWindows()
