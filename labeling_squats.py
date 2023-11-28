import cv2
import os


def extract_and_label_frames(video_path, labels_with_timestamps, output_dir):
    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    current_frame = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_time = current_frame / frame_rate
        label = determine_label_for_time(current_time, labels_with_timestamps)

        # Create a subdirectory for each label if it doesn't exist
        label_dir = os.path.join(output_dir, label)
        os.makedirs(label_dir, exist_ok=True)

        # Save the frame in the appropriate subdirectory
        frame_name = f"frame_{current_frame}.jpg"
        cv2.imwrite(os.path.join(label_dir, frame_name), frame)

        current_frame += 1

    cap.release()


def determine_label_for_time(current_time, labels_with_timestamps):
    for label, time_range in labels_with_timestamps.items():
        if time_range[0] <= current_time <= time_range[1]:
            return label
    return "unknown"  # Default label if no specific label found


# Define your timestamps here (start_time, end_time) in seconds
bad_squat_timestamps = {
    "knees_caving_in": (0, 5),
    "foot_straight": (6, 11),
    "hip_misalignment": (17, 24),
    "knees_over_toes": (25, 29),
    "knees_shaking": (36, 39)
    # Add more as needed
}

good_squat_timestamps = {"good_form": (0, 19)}

# Timestamps and paths for your videos
# Define timestamps (start_time, end_time) in seconds for each video
timestamps = {
    "good_front": {"good_form": (0, 19)},
    "good_side": {"good_form": (0, 27)},
    "bad_front": {
        "knees_caving_in": (0, 5),
        "foot_straight": (6, 11),
        "hip_misalignment": (17, 24),
        "knees_over_toes": (25, 29),
        "knees_shaking": (36, 39)
        # Add more faults as needed
    },
    "bad_side": {
        "knees_caving_in": (5, 13),
        "feet_too_wide": (14, 16),
        "knees_over_toes": (21, 23),
        "using_back": (31, 33)
        # Add more faults as needed
    },
}


video_paths = {
    "good_front": "/Users/yskakshiyap/Desktop/ai-motion-tracker/good squats/IMG_8113 3.MOV",
    "good_side": "/Users/yskakshiyap/Desktop/ai-motion-tracker/good squats/IMG_8112 3.MOV",
    "bad_front": "/Users/yskakshiyap/Desktop/ai-motion-tracker/bad squats/IMG_8113 4.MOV",
    "bad_side": "/Users/yskakshiyap/Desktop/ai-motion-tracker/bad squats/IMG_8112 4.MOV",
}

output_dirs = {
    "good_front": "/Users/yskakshiyap/Desktop/ai-motion-tracker/output/good_front",
    "good_side": "/Users/yskakshiyap/Desktop/ai-motion-tracker/output/good_side",
    "bad_front": "/Users/yskakshiyap/Desktop/ai-motion-tracker/output/bad_front",
    "bad_side": "/Users/yskakshiyap/Desktop/ai-motion-tracker/output/bad_side",
}

# Create output directories if they don't exist
for dir in output_dirs.values():
    os.makedirs(dir, exist_ok=True)

# Process each video
for video_key in video_paths:
    extract_and_label_frames(
        video_paths[video_key], timestamps[video_key], output_dirs[video_key]
    )
