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

labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}

def recognize_sign():
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()

    if not ret:
        app.after(10, recognize_sign)  
        return

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  
                hand_landmarks,  
                mp_hands.HAND_CONNECTIONS,  
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10

        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        try:
            prediction = model.predict([np.asarray(data_aux[:42])])  
        except ValueError:
            prediction = [0] 

        predicted_character = labels_dict[int(prediction[0])]

        letter_var.set(predicted_character)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

    
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = ImageTk.PhotoImage(image=img)

    video_label.img = img  
    video_label.configure(image=img)
    video_label.update()

    app.after(10, recognize_sign)  

def clear_text():
    previous_letter = letter_var.get()
    letter_var.set("")
    previous_letter_var.set(previous_letter)

app = tk.Tk()
app.title("Hand Sign Recognition")


video_frame = tk.Frame(app, relief=tk.SUNKEN, borderwidth=2)
video_frame.pack(side=tk.LEFT, padx=10, pady=10)
video_label = tk.Label(video_frame)
video_label.pack()


letter_frame = tk.Frame(app, relief=tk.RAISED, borderwidth=2)
letter_frame.pack(side=tk.RIGHT, padx=10, pady=10)
previous_letter_var = tk.StringVar()
previous_letter_label = tk.Label(letter_frame, textvariable=previous_letter_var, font=("Helvetica", 36))
previous_letter_label.pack()
letter_var = tk.StringVar()
letter_label = tk.Label(letter_frame, textvariable=letter_var, font=("Helvetica", 36), width=10)  
letter_label.pack()
clear_button = tk.Button(letter_frame, text="Clear", command=clear_text)
clear_button.pack()


cap = cv2.VideoCapture(0)


hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

recognize_sign()  

app.mainloop()

cap.release()
cv2.destroyAllWindows()
