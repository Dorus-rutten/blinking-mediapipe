import cv2
import numpy as np
import mediapipe as mp
import threading

# Constants for facial landmarks
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
Nose = 1
Left_Eye = 33
Right_Eye = 133
Chin = 152

# Blink counters for each camera
blink_counter_1 = 0
blink_counter_2 = 0
best_pitch = 0.4

# Function to calculate EAR
def calculate_ear(eye_landmarks):
    vertical_1 = np.linalg.norm(np.array(eye_landmarks[1]) - np.array(eye_landmarks[5]))
    vertical_2 = np.linalg.norm(np.array(eye_landmarks[2]) - np.array(eye_landmarks[4]))
    horizontal = np.linalg.norm(np.array(eye_landmarks[0]) - np.array(eye_landmarks[3]))
    return (vertical_1 + vertical_2) / (2.0 * horizontal)

# Function to calculate head pose
def get_head_pose(landmarks):
    nose = np.array([landmarks[Nose].x, landmarks[Nose].y, landmarks[Nose].z])
    left_eye = np.array([landmarks[Left_Eye].x, landmarks[Left_Eye].y, landmarks[Left_Eye].z])
    right_eye = np.array([landmarks[Right_Eye].x, landmarks[Right_Eye].y, landmarks[Right_Eye].z])
    chin = np.array([landmarks[Chin].x, landmarks[Chin].y, landmarks[Chin].z])

    nose_chin_vector = chin - nose
    eye_vector = right_eye - left_eye
    nose_chin_vector /= np.linalg.norm(nose_chin_vector)
    eye_vector /= np.linalg.norm(eye_vector)

    pitch = np.arctan2(nose_chin_vector[2], np.linalg.norm(nose_chin_vector[:2]))
    yaw = np.arctan2(eye_vector[1], eye_vector[0])
    roll = np.arctan2(nose_chin_vector[1], eye_vector[0])
    return pitch, yaw, roll

# Adjust RAC based on pitch
def adjust_rac_based_on_pitch(pitch, best_pitch):
    base_rac = 0.10
    scale = 0.02
    if abs(pitch - best_pitch) < 0.10:
        rac = base_rac
    else:
        rac = base_rac - (abs(pitch - best_pitch) * scale)

    # if abs(pitch - best_pitch) >= 0.40:
    #     print("data not trusted")
    #     rac = 0
    # else:
    #     # print(pp)
    #     pass
    return rac

# Function to process each camera
def process_camera(camera_id, window_name, blink_counter):
    cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_id}")
        return

    # Create a separate Mediapipe FaceMesh instance
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    )

    blink_detected = False

    while True:
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

                # Get head pose
                pitch, yaw, roll = get_head_pose(face_landmarks.landmark)

                # Display pitch, yaw, roll
                cv2.putText(frame, f"Pitch: {pitch:.2f}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Yaw: {yaw:.2f}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Roll: {roll:.2f}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Get eye landmarks
                left_eye = [landmarks[i] for i in LEFT_EYE]
                right_eye = [landmarks[i] for i in RIGHT_EYE]

                # Calculate EAR
                left_ear = calculate_ear(left_eye)
                right_ear = calculate_ear(right_eye)
                avg_ear = (left_ear + right_ear) / 2.0

                # Adjust RAC based on pitch
                # if cv2.waitKey(5) & 0xFF == ord('c'):
                #     try:
                #         best_pitch = input("best pitch> ")
                #         best_pitch = float(best_pitch)
                #     except Exception as e:
                #         print(e)
                rac = adjust_rac_based_on_pitch(pitch, best_pitch)

                # Blink detection
                if left_ear < rac or right_ear < rac:
                    if not blink_detected:
                        blink_counter += 1
                        blink_detected = True
                        # cv2.putText(frame, "Blink Detected!", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        # cv2.ractangle(frame, (50, 50), (100, 100), (0, 255, 0), 2)
                        print("Blink detected! cam id: ", camera_id, "blink count: ", blink_counter)


                else:
                    blink_detected = False

                # Draw eye landmarks
                for point in left_eye + right_eye:
                    cv2.circle(frame, point, 2, (255, 0, 0), -1)

        # Show the frame
        cv2.imshow(window_name, frame)

        # Exit on 'q' key
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run both cameras in separate threads
thread1 = threading.Thread(target=process_camera, args=(0, "Camera 1", blink_counter_1))
thread2 = threading.Thread(target=process_camera, args=(1, "Camera 2", blink_counter_2))

thread1.start()
thread2.start()

thread1.join()
thread2.join()
