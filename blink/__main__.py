import cv2
import numpy as np
import mediapipe as mp


LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

#counts the times the cde seed u blink
counter = 0

# Define landmarks for face pose calculation
# We use the nose tip, left and right eyes, and chin for pose estimation
Nose = 1
Left_Eye = 33
Right_Eye = 133
Chin = 152

best_pitch = 0.4

# Function to calculate the head pose using the 3D facial landmarks
def get_head_pose(landmarks):
    # Convert the landmarks into numpy arrays for easier math
    nose = np.array([landmarks[Nose].x, landmarks[Nose].y, landmarks[Nose].z])
    left_eye = np.array([landmarks[Left_Eye].x, landmarks[Left_Eye].y, landmarks[Left_Eye].z])
    right_eye = np.array([landmarks[Right_Eye].x, landmarks[Right_Eye].y, landmarks[Right_Eye].z])
    chin = np.array([landmarks[Chin].x, landmarks[Chin].y, landmarks[Chin].z])
    
    # Find the vectors representing the orientation of the head
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
    # print(vertical_1 + vertical_2 / (2.0 * horizontal)) 

    return (vertical_1 + vertical_2) / (2.0 * horizontal)

# Modify the rac calculation based on the pitch
def adjust_rac_based_on_pitch(pitch,best_pitch):
    # Set a base threshold value for rac
    pp = (best_pitch -pitch)
    scale = 0.02
    base_rac = 0.10
    if abs(pitch - best_pitch) < 0.10:#sencetivety of the blink
        rac = base_rac  # This is your desired value for pitch around 0.40
    else:
        # Scale the threshold dynamically based on the pitch
        # For example, you can make it more sensitive to larger pitch changes
        rac = base_rac - (abs(pitch - best_pitch) * scale)
        
    # if abs(pitch - best_pitch) >= 0.40:
    #     print("data not trusted")
    #     rac = 0
    # else:
    #     # print(pp)
    #     pass

    return rac




cap = cv2.VideoCapture(0)


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
            
            # landmarks = face_landmarks.landmark[33]
            # print(landmarks.z * 1000)
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
            # print(left_ear)
            # print(avg_ear)
            
            # Detect blink
            if cv2.waitKey(5) & 0xFF == ord('c'):
                try:
                    best_pitch = input("best pitch> ")
                    best_pitch = float(best_pitch)
                except Exception as e:
                    print(e)
                
            rac = adjust_rac_based_on_pitch(pitch,best_pitch)  # Adjust rac dynamically based on pitch
            
            
            if left_ear < rac or right_ear < rac:  # Threshold for blink
                if not blink_detected:
                    blink_counter += 1
                    blink_detected = True
                    cv2.putText(frame, "Blink Detected!", (100, 500), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    print(blink_counter)
                    print("left eye is:",left_ear)
                    print("right eye is:",right_ear)
                    if left_ear < rac and right_ear < rac:
                        print("bouht eye are closed")
                    elif left_ear <= rac:
                        print("left eye is closed")

                    elif right_ear <= rac:
                        print("right eye is closed")


                    
            else:
                blink_detected = False
            
            # Draw eye landmarks
            for point in left_eye + right_eye:
                cv2.circle(frame, point, 2, (255, 0, 0), -1)
    
    # Show the frame
    cv2.imshow("Eye Blink Detection", frame)
    
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()