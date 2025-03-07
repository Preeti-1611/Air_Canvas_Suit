import cv2
import numpy as np
import mediapipe as mp

# Initialize Mediapipe Hands module
mpHands = mp.solutions.hands
hands = mpHands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)
mpDraw = mp.solutions.drawing_utils

# Drawing variables
brushThickness = 15
eraserThickness = 50
drawColor = (255, 0, 139)  # Default drawing color
xp, yp = 0, 0  # Previous points
imgCanvas = np.zeros((720, 1280, 3), np.uint8)  # Canvas for drawing

# Tool variables
currentTool = "Free Draw"  # Default tool
startPoint = None  # Starting point for shapes (rectangle, circle, line)
undoStack = []  # Stack to store canvas history for undo

# Webcam setup
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

while True:
    # Capture frame
    success, img = cap.read()
    img = cv2.flip(img, 1)  # Mirror the frame

    # Detect hands
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    totalFingers = 0  # Count of fingers raised across both hands

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            lmList = []
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append((cx, cy))

            if lmList:
                x1, y1 = lmList[8]  # Tip of the index finger
                x2, y2 = lmList[12]  # Tip of the middle finger

                # Determine which fingers are up
                fingers = []
                for i in [8, 12, 16, 20]:  # Index, middle, ring, pinky
                    fingers.append(lmList[i][1] < lmList[i - 2][1])  # Finger is up if above the knuckle
                thumbUp = lmList[4][0] > lmList[3][0]  # Thumb is up if x-coordinate is greater (for left hand)
                fingers.append(thumbUp)
                totalFingers += sum(fingers)

                # Check for selection mode: Two fingers up
                if fingers[0] and fingers[1]:  # Selection mode
                    xp, yp = 0, 0
                    if y1 < 100:  # Tool selection bar
                        if 50 < x1 < 150:
                            drawColor = (255, 0, 139)  # Blue
                            currentTool = "Free Draw"
                        elif 200 < x1 < 300:
                            drawColor = (0, 255, 0)  # Green
                            currentTool = "Free Draw"
                        elif 350 < x1 < 450:
                            drawColor = (0, 0, 255)  # Red
                            currentTool = "Free Draw"
                        elif 500 < x1 < 600:
                            drawColor = (0, 0, 0)  # Eraser
                            currentTool = "Free Draw"
                        elif 650 < x1 < 750:
                            currentTool = "Line"
                        elif 800 < x1 < 900:
                            currentTool = "Rectangle"
                        elif 950 < x1 < 1050:
                            currentTool = "Circle"
                        elif 1100 < x1 < 1200:  # Undo button
                            if undoStack:
                                imgCanvas = undoStack.pop()
                        cv2.rectangle(img, (x1 - 25, y1 - 25), (x1 + 25, y1 + 25), drawColor, cv2.FILLED)

                # Check for drawing mode: Index finger up, middle finger down
                elif fingers[0] and not fingers[1]:
                    cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)

                    if currentTool == "Free Draw":
                        if xp == 0 and yp == 0:  # Starting point
                            xp, yp = x1, y1
                        # Draw lines
                        if drawColor == (0, 0, 0):
                            cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                            cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
                        else:
                            cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                            cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)
                        xp, yp = x1, y1

                    elif currentTool in ["Line", "Rectangle", "Circle"]:
                        if startPoint is None:
                            startPoint = (x1, y1)
                        else:
                            imgCopy = imgCanvas.copy()
                            if currentTool == "Line":
                                cv2.line(imgCopy, startPoint, (x1, y1), drawColor, brushThickness)
                            elif currentTool == "Rectangle":
                                cv2.rectangle(imgCopy, startPoint, (x1, y1), drawColor, brushThickness)
                            elif currentTool == "Circle":
                                radius = int(((startPoint[0] - x1) * 2 + (startPoint[1] - y1) * 2) ** 0.5)
                                cv2.circle(imgCopy, startPoint, radius, drawColor, brushThickness)
                            cv2.imshow("Air Canvas", imgCopy)

                elif not fingers[0]:  # Finalize shapes
                    if currentTool in ["Line", "Rectangle", "Circle"] and startPoint is not None:
                        undoStack.append(imgCanvas.copy())  # Store current canvas state in undoStack
                        if currentTool == "Line":
                            cv2.line(imgCanvas, startPoint, (x1, y1), drawColor, brushThickness)
                        elif currentTool == "Rectangle":
                            cv2.rectangle(imgCanvas, startPoint, (x1, y1), drawColor, brushThickness)
                        elif currentTool == "Circle":
                            radius = int(((startPoint[0] - x1) * 2 + (startPoint[1] - y1) * 2) ** 0.5)
                            cv2.circle(imgCanvas, startPoint, radius, drawColor, brushThickness)
                        startPoint = None
                        # Save the canvas state before drawing the shape
                        undoStack.append(imgCanvas.copy())

    # If all ten fingers are raised, exit the program
    if totalFingers == 10:
        print("Ten-finger gesture detected. Terminating program.")
        break

    # Combine the original frame and the canvas
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)

    # Create toolbar
    cv2.rectangle(img, (50, 1), (150, 100), (255, 0, 0), cv2.FILLED)
    cv2.putText(img, "Blue", (60, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

    cv2.rectangle(img, (200, 1), (300, 100), (0, 255, 0), cv2.FILLED)
    cv2.putText(img, "Green", (210, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

    cv2.rectangle(img, (350, 1), (450, 100), (0, 0, 255), cv2.FILLED)
    cv2.putText(img, "Red", (360, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

    cv2.rectangle(img, (500, 1), (600, 100), (0, 0, 0), cv2.FILLED)
    cv2.putText(img, "Eraser", (510, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

    cv2.rectangle(img, (650, 1), (750, 100), (200, 200, 200), cv2.FILLED)
    cv2.putText(img, "Line", (670, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)

    cv2.rectangle(img, (800, 1), (900, 100), (200, 200, 200), cv2.FILLED)
    cv2.putText(img, "Rect", (820, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)

    cv2.rectangle(img, (950, 1), (1050, 100), (200, 200, 200), cv2.FILLED)
    cv2.putText(img, "Circle", (960, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)

    cv2.rectangle(img, (1100, 1), (1200, 100), (100, 100, 255), cv2.FILLED)
    cv2.putText(img, "Undo", (1110, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

    cv2.imshow("Air Canvas", img)
    cv2.imshow("Canvas", imgCanvas)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()