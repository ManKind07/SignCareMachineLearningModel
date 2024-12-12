import os
import pickle
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3, max_num_hands=2)

DATA_DIR = './data'

data = []
labels = []

# Loop through each class folder and process the images
for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []

        x_ = []
        y_ = []

        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)

        # If multiple hands are detected
        if results.multi_hand_landmarks:
            hands_data = []  # Store both hands' data

            # Loop through all detected hands
            for hand_landmarks in results.multi_hand_landmarks:
                hand_data = []  # Store one hand's data (normalized)

                # Get x, y coordinates for landmarks
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)

                # Normalize and append data for this hand
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    hand_data.append(x - min(x_))  # Normalize x
                    hand_data.append(y - min(y_))  # Normalize y

                hands_data.append(hand_data)

            # If two hands are detected, combine their data
            if len(hands_data) == 2:
                data_aux.extend(hands_data[0])  # First hand
                data_aux.extend(hands_data[1])  # Second hand
            elif len(hands_data) == 1:  # If only one hand is detected, pad the second hand
                data_aux.extend(hands_data[0])  # First hand
                data_aux.extend([0] * len(hands_data[0]))  # Padding for second hand

            # Add the processed data and corresponding label
            data.append(data_aux)
            labels.append(dir_)

# Save the processed data and labels
f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()
