# STEP 1: Import the necessary modules.
import asyncio
import numpy as np
import cv2
import time

from contextlib import suppress
from bleak import BleakScanner, BleakClient


PYBRICKS_COMMAND_EVENT_CHAR_UUID = "c5f50002-8280-46da-89f4-6d8051e4aeef"

# Replace this with the name of your hub if you changed
# it when installing the Pybricks firmware.
HUB_NAME = "Pybricks1"


async def main():
    main_task = asyncio.current_task()

    def handle_disconnect(_):
        print("Hub was disconnected.")

        # If the hub disconnects before this program is done,
        # cancel this program so it doesn't get stuck waiting
        # forever.
        if not main_task.done():
            main_task.cancel()

    ready_event = asyncio.Event()

    def handle_rx(_, data: bytearray):
        if data[0] == 0x01:  # "write stdout" event (0x01)
            payload = data[1:]

            if payload == b"rdy":
                ready_event.set()
            else:
                print("Received:", payload)

    # Do a Bluetooth scan to find the hub.
    device = await BleakScanner.find_device_by_name(HUB_NAME)

    if device is None:
        print(f"could not find hub with name: {HUB_NAME}")
        return

    # Connect to the hub.
    async with BleakClient(device, handle_disconnect) as bleak_client:

        # Subscribe to notifications from the hub.
        await bleak_client.start_notify(PYBRICKS_COMMAND_EVENT_CHAR_UUID, handle_rx)

        # Shorthand for sending some data to the hub.
        async def send(data):
            # Wait for hub to say that it is ready to receive data.
            await ready_event.wait()
            # Prepare for the next ready event.
            ready_event.clear()
            # Send the data to the hub.
            await bleak_client.write_gatt_char(
                PYBRICKS_COMMAND_EVENT_CHAR_UUID,
                b"\x06" + data,  # prepend "write stdin" command (0x06)
                response=True
            )

        # Tell user to start program on the hub.
        print("Start the program on the hub now with the button.")
        

        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        LANDMARKS = mp.solutions.pose.PoseLandmark

        # STEP 2: Create an PoseLandmarker object.
        
        base_options = python.BaseOptions(model_asset_path='pose_landmarker_heavy.task')
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=True)
        detector = vision.PoseLandmarker.create_from_options(options)
        
        # BaseOptions = mp.tasks.BaseOptions
        # PoseLandmarker = mp.tasks.vision.PoseLandmarker
        # PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        # VisionRunningMode = mp.tasks.vision.RunningMode

        # # Create a pose landmarker instance with the video mode:
        # options = PoseLandmarkerOptions(
        #     base_options=BaseOptions(model_asset_path='pose_landmarker_heavy.task'),
        #     running_mode=VisionRunningMode.VIDEO)

        # with PoseLandmarker.create_from_options(options) as landmarker:
        # # The landmarker is initialized. Use it here.
        # # ...
                
        
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
            pose = detection_result.pose_world_landmarks
            
            # # Perform pose landmarking on the provided single image.
            # # The pose landmarker must be created with the video mode.
            # detection_result = landmarker.detect_for_video(image, round(time.time() * 1000))
            
            # pose = detection_result.pose_world_landmarks
            
            if len(pose):
                p_wrist = np.array([pose[0][LANDMARKS.RIGHT_WRIST].x, pose[0][LANDMARKS.RIGHT_WRIST].y, pose[0][LANDMARKS.RIGHT_WRIST].z])
                # p_elbow = np.array([pose[0][LANDMARKS.RIGHT_ELBOW].x, pose[0][LANDMARKS.RIGHT_ELBOW].y, pose[0][LANDMARKS.RIGHT_ELBOW].z])
                p_shoulder = np.array([pose[0][LANDMARKS.RIGHT_SHOULDER].x, pose[0][LANDMARKS.RIGHT_SHOULDER].y, pose[0][LANDMARKS.RIGHT_SHOULDER].z])
                
                # p_elbow = np.array([pose[0][LANDMARKS.RIGHT_ELBOW].x, pose[0][LANDMARKS.RIGHT_ELBOW].y])
                # p_wrist = np.array([pose[0][LANDMARKS.RIGHT_WRIST].x, pose[0][LANDMARKS.RIGHT_WRIST].y])
                # p_shoulder = np.array([pose[0][LANDMARKS.RIGHT_SHOULDER].x, pose[0][LANDMARKS.RIGHT_SHOULDER].y])
                # p_hip = np.array([pose[0][LANDMARKS.RIGHT_HIP].x, pose[0][LANDMARKS.RIGHT_HIP].y])
                
                v_shoulder2wrist = p_wrist - p_shoulder
                v_shoulder2hip = np.array([0, 1, 0]) - p_shoulder
                
                
                ang_h = angle_between(v_shoulder2wrist[[0,2]], v_shoulder2hip[[0,2]])
                ang_v = angle_between(v_shoulder2wrist[1:], v_shoulder2hip[1:])
                ang_h_deg, ang_v_deg = round(np.degrees(ang_h)), round(np.degrees(ang_v))
                print(f"angle h: {ang_h_deg}° v: {ang_v_deg}°")

                cmd = "a" + str(-ang_h_deg) + "\n"
                await send(cmd.encode())
                cmd = "b" + str(ang_v_deg) + "\n"
                await send(cmd.encode())
                
            else:
                print("No human found.")


    print("done.")







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



# Run the main async program.
if __name__ == "__main__":
    with suppress(asyncio.CancelledError):
        asyncio.run(main())