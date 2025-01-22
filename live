import cv2
import numpy as np
import mediapipe as mp

# Set width and height of the output screen
frameWidth = 640
frameHeight = 480
cap = cv2.VideoCapture(0)

# Check if the webcam is accessible
if not cap.isOpened():
    print("Error: Unable to access the webcam.")
    exit()

# Set frame width and height
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, 150)  # Set brightness

# Initialize Mediapipe Hand Tracking
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Color values in BGR for fingertip drawing
fingertipColor = (255, 0, 255)  # Purple for fingertip
drawingColor = (51, 153, 255)  # Drawing with a blue color

# List to store drawn points
myPoints = []

# Main loop to capture video and process it
while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture video.")
        break

    imgResult = img.copy()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    # Process detected hand landmarks
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(imgResult, handLms, mpHands.HAND_CONNECTIONS)
            
            # Get position of the index fingertip (landmark ID 8)
            fingertipX, fingertipY = None, None
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                
                if id == 8:  # Index fingertip
                    fingertipX, fingertipY = cx, cy
                    cv2.circle(imgResult, (cx, cy), 15, fingertipColor, cv2.FILLED)

            # Draw on canvas if fingertip is detected
            if fingertipX is not None and fingertipY is not None:
                myPoints.append((fingertipX, fingertipY))
    
    # Limit the number of points stored
    myPoints = myPoints[-1000:]

    # Draw all points as connected lines
    for i in range(1, len(myPoints)):
        cv2.line(imgResult, myPoints[i-1], myPoints[i], drawingColor, thickness=5)

    # Display the result
    cv2.imshow("Result", imgResult)

    # Press 'q' to exit or 'c' to clear canvas
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        myPoints = []

# Release resources
cap.release()
cv2.destroyAllWindows()
