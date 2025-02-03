import cv2
import numpy as np
import mediapipe as mp
from pylsl import StreamInfo, StreamOutlet
from datetime import datetime
import time

LEFT_EYE = [33, 160, 158, 133, 153, 144]  # Landmarks for the left eye
RIGHT_EYE = [362, 385, 387, 263, 373, 380]  # Landmarks for the right eye

# Ooghoek-landmarks
LEFT_EYE_OUTER = 33
LEFT_EYE_INNER = 133
RIGHT_EYE_OUTER = 362
RIGHT_EYE_INNER = 263

# Ooglid-landmarks
LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145
RIGHT_EYE_TOP = 386
RIGHT_EYE_BOTTOM = 374

MAX_BLINK_DURATION = 0.40
MIN_BLINK_DURATION = 0.05

# Counts the times the code sees you blink
counter = 0

# Define landmarks for face pose calculation
# We use the nose tip, left and right eyes, and chin for pose estimation
Nose = 1
Left_Eye = 33
Right_Eye = 133
Chin = 152

# Variables for blink detection
test_active = False
start_time = None
successful_blinks = 0
failed_blinks = 0
total_tests = 0

best_pitch = 0.4  # The best pitch for the user by default is 0.4

# Create an outlet for the blink stream
info = StreamInfo('BlinkStream', 'Markers', 2, 0, 'string', 'blink1234')  # Name, type, channel count, nominal rate (0 = irregular), format, unique ID
outlet = StreamOutlet(info)  # Create the stream

# Function to send a marker to LSL
def sendlsl():
    global counter
    current_time = datetime.now().strftime("%H:%M:%S.%f")[:-4]  # Get the current time
    outlet.push_sample(["blink", current_time])  # Sending as a list
    counter += 1
    print(f"Blink detected! Time: {current_time} {counter}")

def calculate_equator(landmarks, eye_outer, eye_inner):
    outer = np.array([landmarks[eye_outer].x, landmarks[eye_outer].y])
    inner = np.array([landmarks[eye_inner].x, landmarks[eye_inner].y])
    return (outer + inner) / 2

def detect_blink(landmarks):
    left_equator = calculate_equator(landmarks, LEFT_EYE_OUTER, LEFT_EYE_INNER)
    right_equator = calculate_equator(landmarks, RIGHT_EYE_OUTER, RIGHT_EYE_INNER)
    
    left_top = np.array([landmarks[LEFT_EYE_TOP].x, landmarks[LEFT_EYE_TOP].y])
    left_bottom = np.array([landmarks[LEFT_EYE_BOTTOM].x, landmarks[LEFT_EYE_BOTTOM].y])
    right_top = np.array([landmarks[RIGHT_EYE_TOP].x, landmarks[RIGHT_EYE_TOP].y])
    right_bottom = np.array([landmarks[RIGHT_EYE_BOTTOM].x, landmarks[RIGHT_EYE_BOTTOM].y])
    
    left_blink = left_bottom[1] > left_equator[1] and left_top[1] < left_equator[1]
    right_blink = right_bottom[1] > right_equator[1] and right_top[1] < right_equator[1]
    return left_blink and right_blink


def get_head_pose(landmarks):  # Function to get the head pose
    # Convert the landmarks into numpy arrays for easier math
    nose = np.array([landmarks[Nose].x, landmarks[Nose].y, landmarks[Nose].z])
    left_eye = np.array([landmarks[Left_Eye].x, landmarks[Left_Eye].y, landmarks[Left_Eye].z])
    right_eye = np.array([landmarks[Right_Eye].x, landmarks[Right_Eye].y, landmarks[Right_Eye].z])
    chin = np.array([landmarks[Chin].x, landmarks[Chin].y, landmarks[Chin].z])

    # Calculate the vector between nose and chin
    nose_chin_vector = chin - nose
    # Calculate the vector between left and right eyes
    eye_vector = right_eye - left_eye

    # Normalize vectors
    nose_chin_vector /= np.linalg.norm(nose_chin_vector)
    eye_vector /= np.linalg.norm(eye_vector)

    # Use the cross product to calculate the roll angle
    roll = np.arctan2(nose_chin_vector[1], eye_vector[0])

    # Calculate pitch and yaw by using the vertical and horizontal axis of the head
    pitch = np.arctan2(nose_chin_vector[2], np.linalg.norm(nose_chin_vector[:2]))
    yaw = np.arctan2(eye_vector[1], eye_vector[0])

    return pitch, yaw, roll

def calculate_ear(eye_landmarks):
    # Calculate distances for EAR
    vertical_1 = np.linalg.norm(np.array(eye_landmarks[1]) - np.array(eye_landmarks[5]))
    vertical_2 = np.linalg.norm(np.array(eye_landmarks[2]) - np.array(eye_landmarks[4]))
    horizontal = np.linalg.norm(np.array(eye_landmarks[0]) - np.array(eye_landmarks[3]))
    return (vertical_1 + vertical_2) / (2.0 * horizontal)

# Modify the rac calculation based on the pitch
def adjust_rac_based_on_pitch(pitch, best_pitch):
    # Set a base threshold value for rac
    pp = (best_pitch - pitch)
    scale = 0.02
    base_rac = 0.10
    if abs(pitch - best_pitch) < 0.10:  # Sensitivity of the blink
        rac = base_rac  # This is your desired value for pitch
    else:
        # Scale the threshold dynamically based on the pitch
        # The more the pitch deviates from the desired value, the more the threshold will be
        rac = base_rac - (abs(pitch - best_pitch) * scale)

    # if abs(pitch - best_pitch) >= 0.40:
    #     print("Data not trusted"
    #     rac = 0
    # return rac


cv2.namedWindow("Eye Blink Detection", cv2.WND_PROP_FULLSCREEN)
cap = cv2.VideoCapture(1)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

blink_counter = 0
blink_detected = False
landmark = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip and convert to RGB
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame
    results = face_mesh.process(rgb_frame)


    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            height, width, _ = frame.shape
            landmarks = [(int(lm.x * width), int(lm.y * height)) for lm in face_landmarks.landmark]

            # Get the head pose (pitch, yaw, roll)
            pitch, yaw, roll = get_head_pose(face_landmarks.landmark)

            # Display the pitch, yaw, and roll
            cv2.putText(frame, f"Pitch: {pitch:.2f}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Yaw: {yaw:.2f}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Roll: {roll:.2f}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Optional: Draw landmarks for visualization
            for landmark in [Nose, Left_Eye, Right_Eye, Chin]:
                cv2.circle(frame, landmarks[landmark], 2, (0, 0, 255), -1)

            # Get eye landmarks
            left_eye = [landmarks[i] for i in LEFT_EYE]
            right_eye = [landmarks[i] for i in RIGHT_EYE]

            # Calculate EAR
            left_ear = calculate_ear(left_eye)
            right_ear = calculate_ear(right_eye)

            avg_ear = (left_ear + right_ear) / 2.0

            # Detect blink
            if cv2.waitKey(5) & 0xFF == ord('c'):
                try:
                    best_pitch = float(input("best pitch> "))
                except Exception as e:
                    print(e)

            rac = adjust_rac_based_on_pitch(pitch, best_pitch)  # Adjust rac dynamically based on pitch

            # if left_ear < rac or right_ear < rac:  # Threshold for blink
            #     if not blink_detected:
            #         blink_counter += 1
            #         blink_detected = True
            #         # Send a marker to LSL
            #         sendlsl()
            # else:
            #     blink_detected = False
            
            if detect_blink(face_landmarks.landmark) == False:
                if not blink_detected:
                    blink_start_time = time.time()  # Start the timer when the blink is detected
                    blink_detected = True

            else:
                if blink_detected:
                    blink_duration = time.time() - blink_start_time  # Calculate the blink duration
                    if MIN_BLINK_DURATION <= blink_duration <= MAX_BLINK_DURATION:
                        blink_counter += 1
                        # Send a marker to LSL
                        sendlsl()
                        print(f"Blink duration: {blink_duration:.2f}s (valid)")
                    elif blink_duration < MIN_BLINK_DURATION:
                        print(f"Blink duration: {blink_duration:.2f}s (too short, ignored)")
                    elif blink_duration > MAX_BLINK_DURATION:
                        print(f"Blink duration: {blink_duration:.2f}s (too long, ignored)")
                    blink_detected = False

            # Draw eye landmarks
            for point in left_eye + right_eye:
                cv2.circle(frame, point, 2, (255, 0, 0), -1)

    if test_active:
        # Display the "Blink Now!" text
        cv2.putText(frame, "Blink Now!", (frame.shape[1] // 2 - 100, frame.shape[0] // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

        # Start the timer for the test
        if start_time is None:
            start_time = time.time()

        # If the time has elapsed (2 seconds)
        if time.time() - start_time > 2:  # 2 seconds wait
            # Check the number of blinks within the 2 seconds
            if blink_counter >= 2:  # If more than 3 blinks in 2 seconds, mark as failed
                failed_blinks += 1
                cv2.putText(frame, "Blink Failed! (Too many blinks)", (frame.shape[1] // 2 - 150, frame.shape[0] // 2 + 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            elif blink_counter == 0:  # If no blink was detected within 2 seconds
                failed_blinks += 1
                cv2.putText(frame, "Blink Failed! (No blink detected)", (frame.shape[1] // 2 - 150, frame.shape[0] // 2 + 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                successful_blinks += 1
                cv2.putText(frame, "Blink Success!", (frame.shape[1] // 2 - 100, frame.shape[0] // 2 + 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Set test_active to False after the test
            test_active = False

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s') and test_active == False:  # Start the test
        print("Test started!")
        test_active = True
        total_tests += 1
        blink_counter = 0  # Reset blink count for new test
        start_time = None  # Reset start time
    
    elif key == ord('q'):  # Quit the application
        print(f"Successful blinks: {successful_blinks}")
        print(f"Failed blinks: {failed_blinks}")
        print(f"Total tests: {total_tests}")
        print(f"Success rate: {successful_blinks / total_tests * 100}")
        print("Closing...")
        break

    key = cv2.waitKey(1) & 0xFF
    if key == ord('m'):

        cv2.setWindowProperty("Eye Blink Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
    # Reset for the next test
    if not test_active and key == ord('s'):
        blink_counter = 0  # Reset blink count for the next test
        start_time = None  # Reset start time
        time.sleep(2)  # Wait 2 seconds for the next test

    cv2.imshow("Eye Blink Detection", frame)

cap.release()
cv2.destroyAllWindows()

