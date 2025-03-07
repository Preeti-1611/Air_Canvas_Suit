from cvzone.HandTrackingModule import HandDetector
import cv2
import os
import numpy as np

# Parameters
width, height = 1280, 720
gestureThreshold = 300  # The threshold line for hand gesture (e.g., for navigating slides)
folderPath = "Presentation"  # Folder where the presentation slides are stored

# Camera Setup
cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

# Hand Detector (cvzone)
detectorHand = HandDetector(detectionCon=0.8, maxHands=1)

# Variables
imgList = []
delay = 30  # Delay for button press reset
buttonPressed = False  # Flag for button press state
counter = 0  # Counter for the delay
drawMode = False  # If drawing mode is active
imgNumber = 0  # Image number for current slide
delayCounter = 0
annotations = [[]]  # List of annotations (drawings) per slide
annotationNumber = -1  # Index for the current annotation
annotationStart = False  # Flag to start annotations
hs, ws = int(120 * 1), int(213 * 1)  # width and height of small image for the slide preview

# Get list of presentation images
pathImages = sorted(os.listdir(folderPath), key=len)
print(pathImages)

while True:
    # Get image frame
    success, img = cap.read()
    img = cv2.flip(img, 1)  # Flip the image to mirror for better user experience

    # Load the current slide image
    pathFullImage = os.path.join(folderPath, pathImages[imgNumber])
    imgCurrent = cv2.imread(pathFullImage)

    # Find the hand and its landmarks
    hands, img = detectorHand.findHands(img)  # Detect hand landmarks

    # Draw Gesture Threshold line (for hand gesture control)
    cv2.line(img, (0, gestureThreshold), (width, gestureThreshold), (0, 255, 0), 10)

    if hands and buttonPressed is False:  # If hand is detected and no button pressed
        hand = hands[0]
        cx, cy = hand["center"]  # Hand center coordinates
        lmList = hand["lmList"]  # List of 21 Landmark points
        fingers = detectorHand.fingersUp(hand)  # List indicating which fingers are up

        # Constrain the values for easier drawing
        xVal = int(np.interp(lmList[8][0], [width // 2, width], [0, width]))  # x coordinate of index finger
        yVal = int(np.interp(lmList[8][1], [150, height - 150], [0, height]))  # y coordinate of index finger
        indexFinger = xVal, yVal

        # Terminate the program if all five fingers are up
        if fingers == [1, 1, 1, 1, 1]:
            print("Terminate Program")
            break

        if cy <= gestureThreshold:  # If hand is above the gesture threshold
            # Detect gestures for left and right navigation
            if fingers == [1, 0, 0, 0, 0]:  # Left gesture
                print("Left")
                buttonPressed = True
                if imgNumber > 0:
                    imgNumber -= 1
                    annotations = [[]]  # Clear annotations for new slide
                    annotationNumber = -1
                    annotationStart = False
            if fingers == [0, 0, 0, 0, 1]:  # Right gesture
                print("Right")
                buttonPressed = True
                if imgNumber < len(pathImages) - 1:
                    imgNumber += 1
                    annotations = [[]]  # Clear annotations for new slide
                    annotationNumber = -1
                    annotationStart = False

        # If index finger and middle finger are up, start drawing (annotation mode)
        if fingers == [0, 1, 1, 0, 0]:
            cv2.circle(imgCurrent, indexFinger, 12, (0, 0, 255), cv2.FILLED)  # Draw red circle

        # If only the index finger is up, start or continue annotating
        if fingers == [0, 1, 0, 0, 0]:
            if annotationStart is False:
                annotationStart = True
                annotationNumber += 1
                annotations.append([])  # Add a new list for annotations
            annotations[annotationNumber].append(indexFinger)  # Add annotation to the current slide
            cv2.circle(imgCurrent, indexFinger, 12, (0, 0, 255), cv2.FILLED)  # Draw red circle

        else:
            annotationStart = False  # Stop annotation when no longer drawing

        # If the gesture is to remove the last annotation, pop the last one
        if fingers == [0, 1, 1, 1, 0]:
            if annotations:
                annotations.pop(-1)  # Remove the last annotation
                annotationNumber -= 1
                buttonPressed = True

    else:
        annotationStart = False  # Reset annotation when no hand is detected

    # Handle the button press timeout
    if buttonPressed:
        counter += 1
        if counter > delay:
            counter = 0
            buttonPressed = False

    # Draw annotations (if any)
    for i, annotation in enumerate(annotations):
        for j in range(1, len(annotation)):
            cv2.line(imgCurrent, annotation[j - 1], annotation[j], (0, 0, 200), 12)

    # Display the small preview of the current slide at the top right corner
    imgSmall = cv2.resize(img, (ws, hs))
    h, w, _ = imgCurrent.shape
    imgCurrent[0:hs, w - ws: w] = imgSmall

    # Show the current slide and hand gesture detection
    cv2.imshow("Slides", imgCurrent)
    cv2.imshow("Image", img)

    # Wait for the user to press 'q' to quit
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()