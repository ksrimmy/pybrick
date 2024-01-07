# STEP 1: Import the necessary modules.
import mediapipe as mp
import cv2
import numpy as np



from mediapipe.tasks import python
from mediapipe.tasks.python import vision

LANDMARKS = mp.solutions.pose.PoseLandmark

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))



# STEP 2: Create an PoseLandmarker object.
base_options = python.BaseOptions(model_asset_path='pose_landmarker_heavy.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=True)
detector = vision.PoseLandmarker.create_from_options(options)

# STEP 3: Load the input image.
# image = mp.Image.create_from_file("90.jpg")


# define a video capture object 
vid = cv2.VideoCapture(0) 
  
while(True): 
      
    # Capture the video frame 
    # by frame c
    ret, frame = vid.read() 
    cv2.imshow('my webcam', frame)
    cv2.namedWindow('my webcam',cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('my webcam', width, height)
    if cv2.waitKey(1) == 27: 
        break  # esc to quit
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

    # STEP 4: Detect pose landmarks from the input image.
    detection_result = detector.detect(image)
    pose = detection_result.pose_landmarks
    # pose = detection_result.pose_world_landmarks

    if len(pose):
        p_wrist = np.array([pose[0][LANDMARKS.RIGHT_WRIST].x, pose[0][LANDMARKS.RIGHT_WRIST].y])
        p_elbow = np.array([pose[0][LANDMARKS.RIGHT_ELBOW].x, pose[0][LANDMARKS.RIGHT_ELBOW].y])
        p_shoulder = np.array([pose[0][LANDMARKS.RIGHT_SHOULDER].x, pose[0][LANDMARKS.RIGHT_SHOULDER].y])
        
        v_elbow2wrist =  p_wrist - p_elbow
        v_elbow2shoulder = p_shoulder - p_elbow
        
        angle = angle_between(v_elbow2wrist, v_elbow2shoulder)
        angle_deg = np.degrees(angle)
        print(f"angle: {angle_deg:0.2f}°")

    else:
        print("No human found.")





