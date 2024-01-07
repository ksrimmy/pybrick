# STEP 1: Import the necessary modules.
import asyncio
import numpy as np
import cv2

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
                p_hip = np.array([pose[0][LANDMARKS.RIGHT_HIP].x, pose[0][LANDMARKS.RIGHT_HIP].y])
                
                v_elbow2wrist =  p_wrist - p_elbow
                v_elbow2shoulder = p_shoulder - p_elbow
                
                v_shoulder2hip = p_hip - p_shoulder
                
                ang_elbow = angle_between(v_elbow2wrist, v_elbow2shoulder)
                ang_elbow_deg = round(np.degrees(ang_elbow))
                print(f"angle elbow: {ang_elbow_deg}°")
                
                ang_shoulder = angle_between(v_shoulder2hip, -v_elbow2shoulder)
                ang_shoulder_deg = round(np.degrees(ang_shoulder))
                print(f"angle shoulder: {ang_shoulder_deg}°")

                cmd = "a" + str(ang_elbow_deg) + "\n"
                await send(cmd.encode())
                cmd = "b" + str(ang_shoulder_deg) + "\n"
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