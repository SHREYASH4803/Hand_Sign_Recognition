# import cv2
# from cvzone.HandTrackingModule import HandDetector
# cap =cv2.VideoCapture(0)
# detector = HandDetector( maxHands=2)

# while True:
#     success,img = cap.read()
#     hands,img=detector.findHands(img)
#     cv2.imshow("iamge",img)
#     cv2.waitKey(1)
import time
import cv2
import mediapipe as mp
cap =cv2.VideoCapture(0)
import mediapipe as mp
imagesFolder = "/image"

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands()
_, frame = cap.read()

h, w, c = frame.shape
image_counter = 1  
capture_interval = 20  # Time interval between captures in seconds
capture_countdown = capture_interval  # Initialize the countdown
while True:
    _, frame = cap.read()
   
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(framergb)
    hand_landmarks = result.multi_hand_landmarks
    if hand_landmarks:
        for handLMs in hand_landmarks:
            x_max = 0
            y_max = 0
            x_min = w
            y_min = h
            for lm in handLMs.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                if x > x_max:
                    x_max = x
                if x < x_min:
                    x_min = x
                if y > y_max:
                    y_max = y
                if y < y_min:
                    y_min = y
                rect_padding = 2  # You can adjust this value to increase or decrease the rectangle size
                x_min -= rect_padding
                y_min -= rect_padding
                x_max += rect_padding
                y_max += rect_padding
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            mp_drawing.draw_landmarks(frame, handLMs, mp_hands.HAND_CONNECTIONS,
                                     landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0,0,255), thickness=3, circle_radius=3),
                                     connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=3))
           
    cv2.putText(frame, f"Capture in {capture_countdown} sec", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("Frame", frame)

    capture_countdown -= 1
    if capture_countdown == 0:
        # Reset the countdown
        capture_countdown = capture_interval
        
        # Capture and save the image with a unique filename
        image_filename = f"captured_image_{image_counter}.jpg"
        cv2.imwrite(image_filename, frame)
        print(f"Image {image_filename} captured!")
        image_counter += 1
          

    #display withe detec
    cv2.imshow('Hand Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()