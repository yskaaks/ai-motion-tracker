import cv2
import mediapipe as mp
import numpy as np


# Function to calculate angles
def calculate_angle(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Mid point
    c = np.array(c)  # End point

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(
        a[1] - b[1], a[0] - b[0]
    )
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


# Curl counter logic for both arms
def curl_counter(landmarks, left_counter, left_stage, right_counter, right_stage):
    # Left arm coordinates
    left_shoulder = [
        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
    ]
    left_elbow = [
        landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
        landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y,
    ]
    left_wrist = [
        landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
        landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y,
    ]

    # Right arm coordinates
    right_shoulder = [
        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y,
    ]
    right_elbow = [
        landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
        landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y,
    ]
    right_wrist = [
        landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
        landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y,
    ]

    # Calculate angle for both arms
    left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
    right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

    # Left arm curl counter logic
    if left_angle > 160:
        left_stage = "down"
    elif left_angle < 30 and left_stage == "down":
        left_stage = "up"
        left_counter += 1

    # Right arm curl counter logic
    if right_angle > 160:
        right_stage = "down"
    elif right_angle < 30 and right_stage == "down":
        right_stage = "up"
        right_counter += 1

    return left_counter, left_stage, right_counter, right_stage, left_angle, right_angle


# 3. Function for Squat Counter
def count_squats(landmarks, squat_counter, squat_stage):
    # Use landmarks for LEFT_HIP, LEFT_KNEE, LEFT_ANKLE (or right side)
    hip = [
        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y,
    ]
    knee = [
        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y,
    ]
    ankle = [
        landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
        landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y,
    ]

    # Calculate angle
    angle = calculate_angle(hip, knee, ankle)

    # Squat counting logic
    if angle > 160:
        squat_stage = "up"
    elif angle < 90 and squat_stage == "up":
        squat_stage = "down"
        squat_counter += 1

    return squat_counter, squat_stage, angle


# Function to get user input for the exercise type
def get_exercise_type():
    print("Select the exercise type:")
    print("1: Bicep Curls")
    print("2: Squats")
    choice = input("Enter your choice (1 or 2): ")
    if choice == "1":
        return "bicep_curls"
    elif choice == "2":
        return "squats"
    else:
        print("Invalid choice. Defaulting to Bicep Curls.")
        return "bicep_curls"


# MediaPipe setup
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# MediaPipe setup
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
cap = cv2.VideoCapture(0)

# Initialize counter and stage variables
# Initialize counter and stage variables
left_curl_counter, right_curl_counter = 0, 0
left_curl_stage, right_curl_stage = None, None
squat_count, squat_stage = 0, None


# Ask user to choose the exercise type
exercise_type = get_exercise_type()

# Setup MediaPipe Pose instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        # Convert frame color to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Process frame with MediaPipe Pose
        results = pose.process(image)

        # Convert frame color back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Curl counting logic for both arms
            (
                left_curl_counter,
                left_curl_stage,
                right_curl_counter,
                right_curl_stage,
                left_angle,
                right_angle,
            ) = curl_counter(
                landmarks,
                left_curl_counter,
                left_curl_stage,
                right_curl_counter,
                right_curl_stage,
            )

            # Squat counting logic
            squat_count, squat_stage, squat_angle = count_squats(
                landmarks, squat_count, squat_stage
            )  # Use the renamed function and variable

            # Visualize angle and counter for curls
            cv2.putText(
                image,
                f"Left Curl Angle: {left_angle}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                image,
                f"Right Curl Angle: {right_angle}",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                image,
                f"Left Curls: {left_curl_counter}",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                image,
                f"Right Curls: {right_curl_counter}",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # Visualize angle and counter for squats
            cv2.putText(
                image,
                f"Squat Angle: {squat_angle}",
                (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                image,
                f"Squats: {squat_count}",
                (10, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        # Render detections
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2),
        )

        cv2.imshow("Mediapipe Feed", image)

        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
