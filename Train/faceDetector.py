# myenv\Scripts\activate

from time import time

import cv2
import cvzone
from cvzone.FaceDetectionModule import FaceDetector

####################################
classID = 0  # 0 is fake and 1 is real
confidence = 0.8 
blurThreshold = 35  # Larger is more focus

save = True
debug = False

####################################
offsetPercentageW = 10
offsetPercentageH = 20
camWidth, camHeight = 640, 480
floatingPoint = 6

####################################
outputFolderPath = 'Dataset/DataCollect'
# Địa chỉ IP của ESP32-CAM
cam_url = 'http://192.168.2.22:81/stream'  # Thay <địa_chỉ_ip_ESP32-CAM> bằng địa chỉ IP thực tế
camWidth = 640
camHeight = 480

# Mở video stream từ ESP32-CAM
cap = cv2.VideoCapture(1)
cap.set(3, camWidth)
cap.set(4, camHeight)

detector = FaceDetector()

while True:
    sucess, img = cap.read()
    imgOut = img.copy()
    img, bboxs = detector.findFaces(img, draw=False)

    listBlur = [] # True False values indicating if the faces are blur or not
    listInfo = [] # The nomalized values and the class name for the label txt file

    if bboxs:
        # bboxInfo - "id","bboxs","score"<"center"
        for bbox in bboxs:
            x, y, w, h = bbox["bbox"]
            score = bbox["score"][0]
            # print(x, y, w, h, score)
        #-------------------- Check score -----------------------
            if score > confidence:
            #------------ Them offset cua faceDetected --------------
                #------------ Width -------------------------
                offsetW = (offsetPercentageW/100)  * w
                x = int(x - offsetW)
                w = int(w + offsetW * 2) 
                #------------ Height ------------------------
                offsetH = (offsetPercentageH/100)  * h
                y = int(y - offsetH * 3)
                h = int(h + offsetH * 3.5)    
                #---------- Neu gia tri nho hon 0-------------
                if x < 0: x = 0
                if y < 0: y = 0
                if w < 0: w = 0
                if h < 0: h = 0

            #------------------ Kiem tra blurriness -------------------
                imgFace = img[y:y + h, x:x + w]
                cv2.imshow("Face", imgFace)
                blurValue = int(cv2.Laplacian(imgFace, cv2.CV_64F).var())

                if blurValue>blurThreshold:
                    listBlur.append(True)
                else:
                    listBlur.append(False)
            #--------------------- Normalize Values -------------------
                ih, iw, _ =img.shape
                xc, yc = x + w / 2, y + h /2 

                xcn, ycn = round(xc / iw, floatingPoint), round(yc / ih, floatingPoint)
                wn, hn = round(w / iw, floatingPoint), round(h / ih, floatingPoint)
                # print(xcn, ycn, wn, hn)
                #---------- Neu gia tri lon hon 1 -------------
                if xcn > 1: xcn = 1
                if ycn > 1: ycn = 1
                if wn > 1: wn = 1
                if hn > 1: hn = 1

                listInfo.append(F"{classID} {xcn} {ycn} {wn} {hn}\n")
            #------------------------ Drawing -------------------------
                cv2.rectangle(imgOut,(x, y, w, h),(255,0,0),3)
                cvzone.putTextRect(imgOut, f'Score: {int(score * 100)}% Blur: {blurValue}', (x, y - 0),
                                    scale = 0.8, thickness = 1)
                if debug:
                    cv2.rectangle(img,(x, y, w, h),(255,0,0),3)
                    cvzone.putTextRect(img, f'Score: {int(score * 100)}% Blur: {blurValue}', (x, y - 0),
                                        scale = 0.8, thickness = 1)

        #-------------------- To save -----------------------
        if save:
            if all(listBlur) and listBlur!=[]:
            #----------------------- save image -----------------------
                timeNow = time()
                timeNow = str(timeNow).split('.')
                timeNow = timeNow[0]+timeNow[1]
                print(timeNow)
                cv2.imwrite(f"{outputFolderPath}/{timeNow}.jpg",img)
            #------------------ save Label text File -------------------
                for info in listInfo:
                    f = open(f"{outputFolderPath}/{timeNow}.txt",'a')
                    f.write(info)
                    f.close()


    #----------------------------------------------------------
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    cv2.imshow('ESP32-CAM', imgOut)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
