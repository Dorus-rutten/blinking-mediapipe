import cv2
import numpy as np
import mediapipe as mp
from pylsl import StreamInfo, StreamOutlet
from datetime import datetime
import time

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Initialize variables
counter = 0
test_active = False
start_time = None
successful_blinks = 0
failed_blinks = 0
total_tests = 0

blink_timer_start = None
best_pitch = 0.4  # Default best pitch

# LSL Stream
info = StreamInfo('BlinkStream', 'Markers', 2, 0, 'string', 'blink1234')
outlet = StreamOutlet(info)

def sendlsl():
    current_time = datetime.now().strftime("%H:%M:%S.%f")[:-4]
    outlet.push_sample(["blink", current_time])
    print(f"Blink detected! Time: {current_time}")

def get_head_pose(landmarks):
    nose = np.array([landmarks[1].x, landmarks[1].y, landmarks[1].z])
    left_eye = np.array([landmarks[33].x, landmarks[33].y, landmarks[33].z])
    right_eye = np.array([landmarks[263].x, landmarks[263].y, landmarks[263].z])
    chin = np.array([landmarks[152].x, landmarks[152].y, landmarks[152].z])

    nose_chin_vector = chin - nose
    eye_vector = right_eye - left_eye

    nose_chin_vector /= np.linalg.norm(nose_chin_vector)
    eye_vector /= np.linalg.norm(eye_vector)

    pitch = np.arctan2(nose_chin_vector[2], np.linalg.norm(nose_chin_vector[:2]))
    yaw = np.arctan2(eye_vector[1], eye_vector[0])
    roll = np.arctan2(nose_chin_vector[1], eye_vector[0])

    return pitch, yaw, roll

def calculate_ear(eye_landmarks):
    vertical_1 = np.linalg.norm(np.array(eye_landmarks[1]) - np.array(eye_landmarks[5]))
    vertical_2 = np.linalg.norm(np.array(eye_landmarks[2]) - np.array(eye_landmarks[4]))
    horizontal = np.linalg.norm(np.array(eye_landmarks[0]) - np.array(eye_landmarks[3]))
    return (vertical_1 + vertical_2) / (2.0 * horizontal)

def adjust_rac_based_on_pitch(pitch, best_pitch):
    base_rac = 0.16  # Slightly higher base threshold for better accuracy
    scale = 0.004    # Reduced scale factor for pitch variations
    if abs(pitch - best_pitch) < 0.10:
        return base_rac
    return max(base_rac - abs(pitch - best_pitch) * scale, 0.12)  # Ensure minimum threshold

cap = cv2.VideoCapture(0)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.7, min_tracking_confidence=0.7)

blink_counter = 0
blink_detected = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            height, width, _ = frame.shape
            landmarks = [(int(lm.x * width), int(lm.y * height)) for lm in face_landmarks.landmark]

            pitch, yaw, roll = get_head_pose(face_landmarks.landmark)
            cv2.putText(frame, f"Pitch: {pitch:.2f}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            left_eye = [landmarks[i] for i in LEFT_EYE]
            right_eye = [landmarks[i] for i in RIGHT_EYE]

            left_ear = calculate_ear(left_eye)
            right_ear = calculate_ear(right_eye)
            avg_ear = (left_ear + right_ear) / 2.0

            rac = adjust_rac_based_on_pitch(pitch, best_pitch)

            if avg_ear < rac:  # Eye aspect ratio below threshold
                if not blink_detected:
                    blink_detected = True

                    if blink_timer_start is None:
                        blink_timer_start = time.time()

            else:  # Eye aspect ratio above threshold
                if blink_detected:
                    blink_detected = False
                    if blink_timer_start:
                        blink_duration = time.time() - blink_timer_start
                        blink_timer_start = None

                        # Validate blink duration
                        if 0.3 <= blink_duration <= 0.5:  # Acceptable blink duration: 300ms to 500ms
                            blink_counter += 1
                            sendlsl()  # Send valid blink to LSL
                            print(f"Valid blink! Duration: {blink_duration:.2f} seconds")
                        else:
                            print(f"Invalid blink. Duration: {blink_duration:.2f} seconds")

            for point in left_eye + right_eye:
                cv2.circle(frame, point, 2, (255, 0, 0), -1)

    if test_active:
        cv2.putText(frame, "Blink Now!", (frame.shape[1] // 2 - 100, frame.shape[0] // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

        if start_time is None:
            start_time = time.time()

        if time.time() - start_time > 2:
            if blink_counter == 1:
                successful_blinks += 1
                cv2.putText(frame, "Blink Success!", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                failed_blinks += 1
                cv2.putText(frame, "Blink Failed!", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            test_active = False

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        test_active = True
        total_tests += 1
        blink_counter = 0
        start_time = None
    elif key == ord('q'):
        break

    cv2.imshow("Eye Blink Detection", frame)

cap.release()
cv2.destroyAllWindows()
