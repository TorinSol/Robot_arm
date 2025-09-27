from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision




"""
INITIALIZATION STUFF
"""
cap = cv2.VideoCapture(0)

base_options = python.BaseOptions(model_asset_path='pose_landmarker.task')
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=VisionRunningMode.VIDEO,
        output_segmentation_masks=True)
detector = PoseLandmarker.create_from_options(options)
"""
END OF INITIALIZATION STUFF
"""



"""
Takes an image and draws the skeleton on the detected persons

:args rgb_image: the image to br processed 
:args detection_result: 

:returns the marked up image
"""

def draw_landmarks_on_image_1(rgb_image, detection_result_func):
  pose_landmarks_list = detection_result_func.pose_landmarks
  annotated_image_func = np.copy(rgb_image)

  # Loop through the detected poses to visualize.

  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image_func,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())
  return annotated_image_func


def draw_landmarks_on_image(rgb_image, detection_result_func):
    pose_landmarks_list = detection_result_func.pose_landmarks
    annotated_image_func = np.copy(rgb_image)
    image_height, image_width, _ = rgb_image.shape

    # Loop through the detected poses to visualize.
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

        # Draw the pose landmarks.
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z)
            for landmark in pose_landmarks
        ])

        # Draw landmarks and connections
        solutions.drawing_utils.draw_landmarks(
            annotated_image_func,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style()
        )

        # ➕ Add index numbers next to each landmark
        for i, landmark in enumerate(pose_landmarks):
            x_px = int(landmark.x * image_width)
            y_px = int(landmark.y * image_height)

            # Draw the index number
            cv2.putText(
                annotated_image_func,
                str(i),
                (x_px + 5, y_px - 5),  # Offset text slightly for visibility
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,                  # Font scale
                (0, 255, 0),          # Green text color
                1,                    # Thickness
                cv2.LINE_AA
            )

    return annotated_image_func


while True:
    ret, frame = cap.read()  # Read a frame
    timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
    rgb_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

    if not ret:  # Check if frame was read successfully
        print("Error: Failed to read frame.")
        break

    # STEP 4: Detect pose landmarks from the input image.
    detection_result = detector.detect_for_video(rgb_frame,int(timestamp_ms))

    # STEP 5: Process the detection result. In this case, visualize it.
    annotated_image = draw_landmarks_on_image(rgb_frame.numpy_view(), detection_result)
    cv2.imshow("Live",cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

    #Segmentation mask viewing
    # segmentation_mask = detection_result.segmentation_masks[0].numpy_view()
    # visualized_mask = np.repeat(segmentation_mask[:, :, np.newaxis], 3, axis=2) * 255
    # cv2.imshow("Mask",visualized_mask)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


