import mediapipe as mp
import cv2
import pyautogui
import time
import math

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Initialize OpenCV
cap = cv2.VideoCapture(0)

# Get screen size for mouse movements
screen_width, screen_height = pyautogui.size()

# Variables to hold previous finger positions for gesture detection
prev_x, prev_y = 0, 0

# Thresholds for swipe, scroll, click detection
SWIPE_THRESHOLD = 50  # Pixel threshold to consider as a swipe
SCROLL_THRESHOLD = 30  # Pixel threshold to consider as a scroll
CLICK_THRESHOLD = 15  # Pixel threshold for click detection
JOIN_THRESHOLD = 25  # Distance threshold to consider thumb and index joined

# Function to calculate distance between two points
def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

# Function to count raised fingers based on landmarks
def count_raised_fingers(hand_landmarks):
    raised_fingers = 0
    # Check fingers raised based on the landmark positions (the tip's y-coordinate should be above the joint's y-coordinate)
    # Index(8), Middle(12), Ring(16), Pinky(20)
    for i in [8, 12, 16, 20]:
        if hand_landmarks.landmark[i].y < hand_landmarks.landmark[i - 2].y:  # Check if tip is above the joint
            raised_fingers += 1
    # Check thumb separately (requires horizontal check)
    if hand_landmarks.landmark[4].x > hand_landmarks.landmark[3].x:  # Right-handed thumb
        raised_fingers += 1
    return raised_fingers

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the image horizontally for better interaction
    frame = cv2.flip(frame, 1)

    # Convert the image to RGB (required by MediaPipe)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and get hand landmarks
    result = hands.process(rgb_frame)

    # Initialize the total fingers raised count
    total_fingers = 0

    if result.multi_hand_landmarks:
        # Iterate through each detected hand
        for hand_landmarks in result.multi_hand_landmarks:
            raised_fingers = count_raised_fingers(hand_landmarks)
            total_fingers += raised_fingers

            # Draw landmarks and track the hand
            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Focus on the index finger, middle finger, and thumb
            index_x, index_y = int(hand_landmarks.landmark[8].x * frame.shape[1]), int(hand_landmarks.landmark[8].y * frame.shape[0])
            middle_x, middle_y = int(hand_landmarks.landmark[12].x * frame.shape[1]), int(hand_landmarks.landmark[12].y * frame.shape[0])
            thumb_x, thumb_y = int(hand_landmarks.landmark[4].x * frame.shape[1]), int(hand_landmarks.landmark[4].y * frame.shape[0])

            # Draw circles on relevant landmarks
            cv2.circle(frame, (index_x, index_y), 5, (0, 255, 0), -1)  # Index finger tip
            cv2.circle(frame, (middle_x, middle_y), 5, (0, 0, 255), -1)  # Middle finger tip
            cv2.circle(frame, (thumb_x, thumb_y), 5, (255, 0, 0), -1)  # Thumb tip

            # Calculate the distance between the index and middle fingers
            distance = calculate_distance(index_x, index_y, middle_x, middle_y)

            # Detect scrolling based on the movement of the index and middle fingers
            if abs(middle_y - index_y) > SCROLL_THRESHOLD:
                if middle_y > index_y:
                    pyautogui.scroll(-20)  # Scroll down
                elif middle_y < index_y:
                    pyautogui.scroll(0)  # Scroll up

            # Detect click gesture (index and thumb join)
            if calculate_distance(index_x, index_y, thumb_x, thumb_y) < JOIN_THRESHOLD:
                pyautogui.click()
                time.sleep(0.2)  # To avoid multiple clicks

            # Map the index finger's position to the screen's coordinates
            screen_x = int(screen_width * hand_landmarks.landmark[8].x)
            screen_y = int(screen_height * hand_landmarks.landmark[8].y)

            # Move the mouse cursor with the index finger
            pyautogui.moveTo(screen_x, screen_y)

            # Detect swipe gesture (horizontal movement of the index finger)
            if abs(index_x - prev_x) > SWIPE_THRESHOLD:
                if index_x > prev_x:
                    pyautogui.press("right")  # Swipe right (right arrow key)
                else:
                    pyautogui.press("left")  # Swipe left (left arrow key)

            # Detect swipe gesture (vertical movement of the index finger)
            if abs(index_y - prev_y) > SWIPE_THRESHOLD:
                if index_y > prev_y:
                    pyautogui.press("down")  # Swipe down (down arrow key)
                else:
                    pyautogui.press("up")  # Swipe up (up arrow key)

            prev_x, prev_y = index_x, index_y

    # Check if five fingers are detected
    if total_fingers >= 8:
        print("Five fingers detected. Terminating program.")
        break

    # Show the resulting frame
    cv2.imshow("Hand Gesture Control", frame)

    # Exit the program when the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
