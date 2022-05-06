import os
import mediapipe as mp
import cv2
import math
import time
from time import sleep

from cvzone.SerialModule import SerialObject 


class HandDetector:
    def __init__(self, maxHands=2, detectionCon=0.5, minTrackCon=0.5):
        """
        maxHands: Maximum number of hands to detect
        detectionCon: Minimum Detection Confidence Threshold
        minTrackCon: Minimum Tracking Confidence Threshold
        """
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.minTrackCon = minTrackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(max_num_hands=self.maxHands,
                                        min_detection_confidence=self.detectionCon,
                                        min_tracking_confidence=self.minTrackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]
        self.fingers = []
        self.lmList = []

    def findHands(self, img, draw=True, flipType=True):
        """
        img: Image to find the hands in.
        draw: Flag to draw the output on the image.
        Image with or without drawings
        """
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        allHands = []
        h, w, c = img.shape
        if self.results.multi_hand_landmarks:
            for handType, handLms in zip(self.results.multi_handedness, self.results.multi_hand_landmarks):
                myHand = {}
                ## lmList
                mylmList = []
                xList = []
                yList = []
                for id, lm in enumerate(handLms.landmark):
                    px, py, pz = int(lm.x * w), int(lm.y * h), int(lm.z * w)
                    mylmList.append([px, py, pz])
                    xList.append(px)
                    yList.append(py)

                ## bbox
                xmin, xmax = min(xList), max(xList)
                ymin, ymax = min(yList), max(yList)
                boxW, boxH = xmax - xmin, ymax - ymin
                bbox = xmin, ymin, boxW, boxH
                cx, cy = bbox[0] + (bbox[2] // 2), \
                         bbox[1] + (bbox[3] // 2)

                myHand["lmList"] = mylmList
                myHand["bbox"] = bbox
                myHand["center"] = (cx, cy)

                if flipType:
                    if handType.classification[0].label == "Right":
                        myHand["type"] = "Left"
                    else:
                        myHand["type"] = "Right"
                else:
                    myHand["type"] = handType.classification[0].label
                allHands.append(myHand)

                ## draw
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)
                    cv2.rectangle(img, (bbox[0] - 20, bbox[1] - 20),
                                  (bbox[0] + bbox[2] + 20, bbox[1] + bbox[3] + 20),
                                  (255, 0, 255), 2)
                    cv2.putText(img, myHand["type"], (bbox[0] - 30, bbox[1] - 30), cv2.FONT_HERSHEY_PLAIN,
                                2, (255, 0, 255), 2)
        if draw:
            return allHands, img
        else:
            return allHands

    def fingersUp(self, myHand):
        """
        Finds how many fingers are open and returns in a list.
        """
        myHandType = myHand["type"]
        myLmList = myHand["lmList"]
        if self.results.multi_hand_landmarks:
            fingers = []
            # Thumb
            if myHandType == "Right":
                if myLmList[self.tipIds[0]][0] > myLmList[self.tipIds[0] - 1][0]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            else:
                if myLmList[self.tipIds[0]][0] < myLmList[self.tipIds[0] - 1][0]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            # 4 Fingers
            for id in range(1, 5):
                if myLmList[self.tipIds[id]][1] < myLmList[self.tipIds[id] - 2][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)
        return fingers
    
    def findDistance(self, p1, p2, img=None):
        """
        Find the distance between two landmarks based on their
        index numbers.
        p1: Point1
        p2: Point2
        img: Image to draw on.
        draw: Flag to draw the output on the image
        """
        x1, y1 = p1
        x2, y2 = p2
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        length = math.hypot(x2 - x1, y2 - y1)
        info = (x1, y1, x2, y2, cx, cy)
        if img is not None:
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
            return length, info, img
        else:
            return length, info

def main():

    arduino = SerialObject(portNo='COM13', baudRate=9600, digits=1)

    folderPath = "Fingers"
    myList = os.listdir(folderPath)
    myList.sort()
    print(myList)

    overlayList = []
    for imPath in myList:
        image = cv2.imread(f'{folderPath}/{imPath}')
        print(f'{folderPath}/{imPath}')
        overlayList.append(image)
    
    cap = cv2.VideoCapture(0)
    detector = HandDetector(detectionCon=0.75, maxHands=2)
    pTime = 0

    while True:
        # Get image frame
        success, img = cap.read()
        # Find the hand and its landmarks
        hands, img = detector.findHands(img)  

        if hands:
            # Hand 1
            hand1 = hands[0]
            # List of 21 Landmarks points (Check Downloaded Image)
            lmList1 = hand1["lmList"]  # List of 21 Landmark points
            bbox1 = hand1["bbox"]  # Bounding box info x,y,w,h
            centerPoint1 = hand1['center']  # center of the hand cx,cy
            handType1 = hand1["type"]  # Handtype Left or Right
            fingers1 = detector.fingersUp(hand1)

            if (handType1 == "Right"):
                for i in range(len(fingers1)):
                    if fingers1[i] == 1:
                        print(i+1)
                        arduino.sendData([i+1])
                        

            if len(hands) == 2:
                hand2 = hands[1]
                lmList2 = hand2["lmList"]  # List of 21 Landmarks points
                bbox2 = hand2["bbox"]  # Bounding Box info x,y,w,h
                centerPoint2 = hand2["center"]  # center of the hand cx,cy
                handType2 = hand2["type"]  # Hand Type Left or Right
                fingers2 = detector.fingersUp(hand2)

                if (handType2 == "Left"):
                    totalFingers = fingers2.count(1)
                    cv2.putText(img, str(totalFingers), (50, 400), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 5)
                    h, w, c = overlayList[totalFingers].shape
                    img[0:h, 0:w] = overlayList[totalFingers]

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, f'FPS: {int(fps)}', (575, 20),
                    cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
        cv2.imshow("Image", img)

        if cv2.waitKey(5) & 0xFF == 27:
            break
    cap.release()

if __name__ == "__main__":
    main()
