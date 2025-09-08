import cv2
import mediapipe as mp
import pandas as pd

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,   # one hand at a time for alphabets
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Parameters
frames_to_capture = 100
a = []  # collected landmarks with labels

# Start video capture
cap = cv2.VideoCapture(0)

# Ask user which alphabet we’re recording
alphabet = input("Enter the alphabet you want to record: ").upper()

print(f"Press 'q' to start recording {frames_to_capture} frames for '{alphabet}'")

recording = False
captured_frames = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') and not recording:
        print("Recording started...")
        recording = True
        captured_frames = 0

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    # Draw landmarks + collect data
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            if recording and captured_frames < frames_to_capture:
                temp = {}
                for i, land in enumerate(hand_landmarks.landmark):
                    temp[i] = [land.x, land.y, land.z]  # normalized coords

                # Flatten into a row
                row = []
                for i in range(21):
                    row.extend(temp[i])  # [x, y, z]
                row.append(alphabet)  # add label

                a.append(row)
                captured_frames += 1
                print(f"Captured frame {captured_frames}/{frames_to_capture}")

                if captured_frames == frames_to_capture:
                    recording = False
                    print(f"Recording for '{alphabet}' finished.")

    # Show video feed
    cv2.imshow("MediaPipe Hands - Dataset Collector", frame)

    if key == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()

# -------- Save to CSV --------
columns = [f"{axis}{i}" for i in range(21) for axis in ["x", "y", "z"]] + ["label"]
df = pd.DataFrame(a, columns=columns)

# Append new data if CSV already exists
csv_file = "hand_sign_dataset.csv"
try:
    old_df = pd.read_csv(csv_file)
    df = pd.concat([old_df, df], ignore_index=True)
except FileNotFoundError:
    pass

df.to_csv(csv_file, index=False)
print(f"Data for '{alphabet}' saved to {csv_file}")
