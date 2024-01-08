import tkinter as tk
import cv2
import mediapipe as mp
import numpy as np
import pickle
from PIL import Image, ImageTk

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S',
    19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'
}

def calculate_average_distance(landmarks_3d):
    num_landmarks = len(landmarks_3d)
    total_distance = sum(calculate_distance(landmarks_3d[i], (0, 0, 0)) for i in range(num_landmarks))
    return total_distance / num_landmarks if num_landmarks > 0 else 0

def calculate_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2 + (point1[2] - point2[2])**2)

def calculate_hand_size(x1, y1, x2, y2):
    return (x2 - x1) * (y2 - y1)

def recognize_sign():
    ret, frame = cap.read()

    if not ret:
        app.after(10, recognize_sign)
        return

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks_3d = [(landmark.x, landmark.y, landmark.z) for landmark in hand_landmarks.landmark]
            average_distance = calculate_average_distance(landmarks_3d)
            max_allowed_distance = 1.5

            # Adjust the confidence threshold
            if results.multi_handedness[0].classification[0].score < 0.3 or average_distance > max_allowed_distance:
                continue

            data_aux = []  
            x_ = []
            y_ = []

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10

            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

           
            hand_size = calculate_hand_size(x1, y1, x2, y2)

           
            hand_size_threshold = 7500

            if hand_size < hand_size_threshold:
                continue

            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            data_aux = []  
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

            try:
                prediction = model.predict([np.asarray(data_aux[:42])])
            except ValueError:
                prediction = [0]

            predicted_character = labels_dict[int(prediction[0])]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = ImageTk.PhotoImage(image=img)

    video_label.img = img
    video_label.configure(image=img)
    video_label.update()

    app.after(10, recognize_sign)

app = tk.Tk()
app.title("Hand Sign Recognition")

video_frame = tk.Frame(app, relief=tk.SUNKEN, borderwidth=2)
video_frame.pack(padx=10, pady=10)
video_label = tk.Label(video_frame)
video_label.pack()

cap = cv2.VideoCapture(0)

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

recognize_sign()

app.mainloop()

cap.release()
cv2.destroyAllWindows()
