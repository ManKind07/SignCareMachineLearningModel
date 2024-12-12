import pickle
import cv2
import mediapipe as mp
import numpy as np
import pyvirtualcam
import logging

# Configure logging to file
logging.basicConfig(filename="debug.log", level=logging.DEBUG, format="%(asctime)s - %(message)s")

try:
    logging.debug("Application started.")

    # Load the trained model
    logging.debug("Loading model...")
    model_dict = pickle.load(open('./model.p', 'rb'))
    model = model_dict['model']
    logging.debug("Model loaded successfully.")

    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3, max_num_hands=2)

    # Labels dictionary (adjust to your custom labels)
    labels_dict = {0: 'Good', 1: 'Morning', 2: 'My', 3: 'Throat hurts', 4: '2', 5: 'days', 6: 'days', 7: 'Yes', 8: 'Thank You', 9: 'Doctor'}

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.error("Webcam not detected.")
        raise Exception("Webcam not detected.")

    logging.debug("Webcam initialized.")

    # Create a virtual camera
    with pyvirtualcam.Camera(width=640, height=480, fps=30) as cam:
        logging.debug(f'Virtual camera created: {cam.device}')

        while True:
            ret, frame = cap.read()
            if not ret:
                logging.warning("Failed to capture frame.")
                continue

            H, W, _ = frame.shape

            # Prepare the frame for processing
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            data_aux = []
            x_ = []
            y_ = []

            if results.multi_hand_landmarks:
                hands_data = []  # List to hold the data for both hands

                for hand_landmarks in results.multi_hand_landmarks:
                    hand_data = []  # Data for the current hand

                    # Extract landmarks
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        x_.append(x)
                        y_.append(y)

                    # Normalize and add the data for this hand
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        hand_data.append(x - min(x_))  # Normalize x
                        hand_data.append(y - min(y_))  # Normalize y

                    hands_data.append(hand_data)

                # If only one hand is detected, pad the second hand's data with zeros
                if len(hands_data) == 1:
                    hands_data.append([0] * len(hands_data[0]))  # Padding for second hand

                # Combine data for both hands (84 features)
                data_aux = hands_data[0] + hands_data[1]

                # Make prediction
                prediction = model.predict([np.asarray(data_aux)])

                # Decode prediction to gesture label
                predicted_character = labels_dict[int(prediction[0])]

                # Draw the landmarks and the predicted gesture
                for hand_landmarks in results.multi_hand_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                    )

                # Display bounding box and prediction
                if hands_data:
                    x1 = int(min(x_) * W) - 10
                    y1 = int(min(y_) * H) - 10
                    x2 = int(max(x_) * W) - 10
                    y2 = int(max(y_) * H) - 10
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)

                    # Place caption at the top-center of the frame
                    text = predicted_character
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                    text_x = (frame.shape[1] - text_size[0]) // 2  # Calculate the center position
                    text_y = 30  # Set the y-position near the top of the frame

                    # Draw the text on the frame in white color with Arial font
                    cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Send the frame to the virtual camera
            cam.send(frame)
            cam.sleep_until_next_frame()

            # Optional: Display frame locally as well
            cv2.imshow('Frame', frame)

            # Break loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release webcam and close OpenCV window
    cap.release()
    cv2.destroyAllWindows()

except Exception as e:
    logging.error(f"An error occurred: {e}")
    input("Press Enter to exit...")  # Keep the terminal open for error inspection
