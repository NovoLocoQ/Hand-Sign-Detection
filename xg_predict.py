import cv2
import mediapipe as mp
import joblib
import numpy as np
import pandas as pd

# Load trained model + label encoder
model = joblib.load("hand_sign_xgb_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Column names must match training data
columns = [f"{axis}{i}" for i in range(21) for axis in ["x", "y", "z"]]

# Start webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    prediction_text = ""

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw landmarks on hand
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract landmarks
            row = []
            for lm in hand_landmarks.landmark:
                row.extend([lm.x, lm.y, lm.z])

            # Convert to DataFrame with correct column names
            X_live = pd.DataFrame([row], columns=columns)

            # Predict with probabilities
            probs = model.predict_proba(X_live)[0]
            pred_idx = np.argmax(probs)
            predicted_label = label_encoder.inverse_transform([pred_idx])[0]
            confidence = probs[pred_idx]

            prediction_text = f"{predicted_label} ({confidence:.2f})"

            # Show prediction on frame
            cv2.putText(frame, prediction_text, (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    # Show window
    cv2.imshow("Hand Sign Detection", frame)

    # Exit with ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
