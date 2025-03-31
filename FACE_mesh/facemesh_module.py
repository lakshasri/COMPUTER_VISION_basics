import cv2
import mediapipe as mp
import time

class FaceMeshDetector:
    def __init__(self,staticMode=False,maxFaces=2,minDetectionCon=0.5,minTrackCon=0.5):
        self.staticMode=staticMode
        self.maxFaces=maxFaces
        self.minDetectionCon=minDetectionCon
        self.minTrackCon=minTrackCon

        self.draw=mp.solutions.drawing_utils
        self.fmesh=mp.solutions.face_mesh
        self.faceMesh=self.fmesh.FaceMesh(self.staticMode,self.maxFaces,self.minDetectionCon,self.minTrackCon)
        self.spec=self.draw.DrawingSpec(thickness=1,circle_radius=2) # just enough detail to look cool

    def findFaceMesh(self,img,draw=True):
        imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        results=self.faceMesh.process(imgRGB)
        faces=[]
        if results.multi_face_landmarks:
            for faceLms in results.multi_face_landmarks:
                if draw:
                    self.draw.draw_landmarks(img,faceLms,self.fmesh.FACEMESH_TESSELATION,self.spec,self.spec) # fancy connect-the-dots
                face=[]
                for id,lm in enumerate(faceLms.landmark):
                    ih,iw,ic=img.shape
                    x,y=int(lm.x*iw),int(lm.y*ih)
                    face.append([x,y])
                faces.append(face)
        return img,faces

def main():
    cap=cv2.VideoCapture("../videos/1.mp4")
    pTime=0
    detector=FaceMeshDetector(maxFaces=2)
    while True:
        success,img=cap.read()
        img,faces=detector.findFaceMesh(img)
        if faces:
            print(faces[0]) # logging only the first face, we don't need an essay
        cTime=time.time()
        fps=1/(cTime-pTime)
        pTime=cTime
        cv2.putText(img,f'FPS: {int(fps)}',(20,70),cv2.FONT_HERSHEY_PLAIN,3,(0,255,0),3) # FPS, because speed matters
        cv2.imshow("Image",img)
        cv2.waitKey(1)

if __name__=="__main__":
    main()
