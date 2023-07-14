import cv2
import mediapipe as mp
import time

# Camera Access
cap = cv2.VideoCapture(0)

# Using hand Detection Module Builtin
mpHands = mp.solutions.hands
hands = mpHands.Hands()  # object of hand
mpDraw = mp.solutions.drawing_utils

cTime = 0
pTime = 0

# Calling Camera
while True:
    success, img = cap.read()

    # Image to RGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)  # Process the frame of hands

    # Extract the multiple Hands
    finger_counts = []  # Store the finger counts for each hand

    if results.multi_hand_landmarks:
        for handlms in results.multi_hand_landmarks:
            finger_count = 0  # Initialize finger count for the current hand

            # Check if thumb is extended
            thumb_extended = handlms.landmark[4].y < handlms.landmark[3].y

            for id, lm in enumerate(handlms.landmark):
                # This gives the each point(id) and their landmark(x,y,z) location
                # x,y,z is ratio, so we convert them into pixels by multiplying with width and height
                h, w, c = img.shape  # c for column
                cx, cy = int(lm.x * w), int(lm.y * h)  # Centre point

                # Controling each id(point)/ Tip of Fingers
                if id == 4:  # Thumb
                    if not thumb_extended:
                        continue  # Skip counting if thumb is not extended
                elif id in [8, 12, 16, 20]:  # Fingers
                    if cy < handlms.landmark[id - 2].y * h:  # Check if the current point is above the previous point
                        finger_count += 1  # Increment finger count for the current hand

                cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            finger_counts.append(finger_count)  # Store the finger count for the current hand

            # Drawing the 21 points with the mediaPipe
            mpDraw.draw_landmarks(img, handlms, mpHands.HAND_CONNECTIONS)

    # Display finger counts for each hand
    for i, count in enumerate(finger_counts):
        y_pos = 70 + i * 30  # Adjust the y-position of the text based on the number of hands detected
        cv2.putText(img, f"Hand {i+1} fingers: {count}", (10, y_pos), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

    # Calculate and display the sum of all fingers for each hand (excluding the thumb)
    total_fingers = sum(finger_counts)
    cv2.putText(img, f"Total fingers (excluding thumb): {total_fingers}", (10, 40), cv2.FONT_HERSHEY_PLAIN, 2,
                (0, 0, 255), 2)

    # FPS Calculation
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f"FPS: {int(fps)}", (10, 400), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
