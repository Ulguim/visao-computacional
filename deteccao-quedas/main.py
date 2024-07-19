import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)

video_path = 'sample.mp4'
cap = cv2.VideoCapture(video_path)

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    rad = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(rad * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

frame_count = 0
fall_detected_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
        )

        landmarks = results.pose_landmarks.landmark
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
        right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]

        height, width, _ = frame.shape
        left_shoulder_coords = (int(left_shoulder.x * width), int(left_shoulder.y * height))
        right_shoulder_coords = (int(right_shoulder.x * width), int(right_shoulder.y * height))
        left_hip_coords = (int(left_hip.x * width), int(left_hip.y * height))
        right_hip_coords = (int(right_hip.x * width), int(right_hip.y * height))
        left_knee_coords = (int(left_knee.x * width), int(left_knee.y * height))
        right_knee_coords = (int(right_knee.x * width), int(right_knee.y * height))

        trunk_angle = calculate_angle(left_shoulder_coords, left_hip_coords, right_shoulder_coords)

        knee_angle = calculate_angle(left_hip_coords, left_knee_coords, right_hip_coords)

        if (trunk_angle < 10 or trunk_angle > 170) and knee_angle < 30:
            attention_status = "Pessoa Caida"
            fall_detected_count += 1
            cv2.imwrite(f'fall_detection_{fall_detected_count}.png', frame)
        elif (trunk_angle > 45 and trunk_angle < 135) and knee_angle > 30:
            attention_status = "Possívelmente Caida"
        else:
            attention_status = "Atenção Normal"

        cv2.putText(frame, attention_status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow('Pose Estimation', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
