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
HUB_NAME = "brick1"


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
                response=True,
            )

        # Tell user to start program on the hub.
        print("Start the program on the hub now with the button.")

        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        LANDMARKS = mp.solutions.pose.PoseLandmark

        # STEP 2: Create an PoseLandmarker object.

        # base_options = python.BaseOptions(model_asset_path='pose_landmarker_heavy.task')
        # options = vision.PoseLandmarkerOptions(
        #     base_options=base_options,
        #     output_segmentation_masks=True)
        # BaseOptions = mp.tasks.BaseOptions

        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        # Create a pose landmarker instance with the video mode:
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path='pose_landmarker_heavy.task'),
            running_mode=VisionRunningMode.VIDEO)

        # # Create a pose landmarker instance with the live stream mode:
        # def print_result(
        #     result: mp.tasks.vision.PoseLandmarkerResult,
        #     output_image: mp.Image,
        #     timestamp_ms: int,
        # ):
        #     print("pose landmarker result: {}".format(result))

        # options = PoseLandmarkerOptions(
        #     base_options=BaseOptions(model_asset_path="pose_landmarker_heavy.task"),
        #     running_mode=VisionRunningMode.LIVE_STREAM,
        #     result_callback=print_result,
        # )
        
        # BaseOptions = mp.tasks.BaseOptions
        # PoseLandmarker = mp.tasks.vision.PoseLandmarker
        # PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        # VisionRunningMode = mp.tasks.vision.RunningMode

        # # Create a pose landmarker instance with the video mode:
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path='pose_landmarker_heavy.task'),
            running_mode=VisionRunningMode.VIDEO)
        
        detector = PoseLandmarker.create_from_options(options)

        # with PoseLandmarker.create_from_options(options) as landmarker:
        # # The landmarker is initialized. Use it here.
        # # ...

        # STEP 3: Load the input image.
        # image = mp.Image.create_from_file("90.jpg")
        # define a video capture object
        vid = cv2.VideoCapture(0)
        buffer_size = 5
        last_values_l, last_values_r = np.zeros(buffer_size), np.zeros(buffer_size)

        while True:

            # Capture the video frame
            # by frame c
            ret, frame = vid.read()
            cv2.imshow("my webcam", cv2.flip(cv2.resize(frame, (1200, 1000)), 1))
            cv2.namedWindow("my webcam", cv2.WINDOW_NORMAL)
            if cv2.waitKey(1) == 27:
                break  # esc to quit
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

            # STEP 4: Detect pose landmarks from the input image.

            # detection_result = detector.detect(image)
            detection_result = detector.detect_for_video(image, round(time.time() * 1000))
            # detection_result = detector.detect_async(image, round(time.time() * 1000))
            pose = detection_result.pose_world_landmarks

            # # Perform pose landmarking on the provided single image.
            # # The pose landmarker must be created with the video mode.
            # detection_result = landmarker.detect_for_video(image, round(time.time() * 1000))

            # pose = detection_result.pose_world_landmarks

            if len(pose):
                p_wrist_r = np.array(
                    [
                        pose[0][LANDMARKS.RIGHT_WRIST].x,
                        pose[0][LANDMARKS.RIGHT_WRIST].y,
                        pose[0][LANDMARKS.RIGHT_WRIST].z,
                    ]
                )
                p_shoulder_r = np.array(
                    [
                        pose[0][LANDMARKS.RIGHT_SHOULDER].x,
                        pose[0][LANDMARKS.RIGHT_SHOULDER].y,
                        pose[0][LANDMARKS.RIGHT_SHOULDER].z,
                    ]
                )

                p_wrist_l = np.array(
                    [
                        pose[0][LANDMARKS.LEFT_WRIST].x,
                        pose[0][LANDMARKS.LEFT_WRIST].y,
                        pose[0][LANDMARKS.LEFT_WRIST].z,
                    ]
                )
                p_shoulder_l = np.array(
                    [
                        pose[0][LANDMARKS.LEFT_SHOULDER].x,
                        pose[0][LANDMARKS.LEFT_SHOULDER].y,
                        pose[0][LANDMARKS.LEFT_SHOULDER].z,
                    ]
                )
                p_eye_l = np.array(
                    [
                        pose[0][LANDMARKS.LEFT_EAR].x,
                        pose[0][LANDMARKS.LEFT_EAR].y,
                        pose[0][LANDMARKS.LEFT_EAR].z,
                    ]
                )
                p_eye_r = np.array(
                    [
                        pose[0][LANDMARKS.RIGHT_EAR].x,
                        pose[0][LANDMARKS.RIGHT_EAR].y,
                        pose[0][LANDMARKS.RIGHT_EAR].z,
                    ]
                )

                v_shoulder2wrist_r = p_wrist_r - p_shoulder_r
                v_shoulder2hip_r = np.array([0, 0, 0]) - p_shoulder_r

                # v_shoulder2shoulder = p_shoulder_l- p_shoulder_r
                v_eye2eye = p_eye_l - p_eye_r
                
                # p_hip_r = np.array(
                #     [
                #         pose[0][LANDMARKS.RIGHT_HIP].x,
                #         pose[0][LANDMARKS.RIGHT_HIP].y,
                #         pose[0][LANDMARKS.RIGHT_HIP].z,
                #     ]
                # )
                # p_hip_l = np.array(
                #     [
                #         pose[0][LANDMARKS.LEFT_HIP].x,
                #         pose[0][LANDMARKS.LEFT_HIP].y,
                #         pose[0][LANDMARKS.LEFT_HIP].z,
                #     ]
                # )
                # v_hip2hip = p_hip_r - p_hip_l

                v_shoulder2wrist_l = p_wrist_l - p_shoulder_l
                v_shoulder2hip_l = np.array([0, 0, 0]) - p_shoulder_l

                # upper body coronal
                # ang_upper_body = cos_angle_between([1,0], v_shoulder2shoulder[[0,2]])
                
                
                # arms
                ang_vl = np.arccos(cos_angle_between(v_shoulder2wrist_l[1:], v_shoulder2hip_l[1:]))
                ang_vr = np.arccos(cos_angle_between(v_shoulder2wrist_r[1:], v_shoulder2hip_r[1:]))
                ang_l_deg, ang_r_deg = np.degrees(ang_vl), np.degrees(ang_vr)
                
                # head coronal
                # cos_ang_eyes = cos_angle_between([1, 0], v_eye2eye[[0, 1]])
                # ang_eyes = np.arccos(cos_ang_eyes)
                # ang_eyes_deg = np.degrees(ang_eyes)
                # ang_eyes_deg = ang_eyes_deg - 20
                # ang_eyes_deg = np.clip(ang_eyes_deg, -30, 30) * -1
                
                # ang_eyes_deg = ang_eyes_deg - 20
                # ang_eyes_deg = np.clip(ang_eyes_deg, -30, 30) * -1
                
                
                # head traversal
                cos_head_trav = cos_angle_between([0, 1], v_eye2eye[[0, 2]])
                head_trav = np.arccos(cos_head_trav)
                head_trav_deg = np.degrees(head_trav)
                head_trav_deg = (head_trav_deg - 90) * -1
                # if cos_head_trav < 0:
                #     print("negative")
                    
                head_trav_deg = np.clip(head_trav_deg, -30, 30)
                
                
                print(
                    f"left: {ang_l_deg:0.1f}° right: {ang_r_deg:0.1f}° head: {head_trav_deg:0.1f}"
                )

                # last_values_l = np.roll(last_values_l, 1)
                # last_values_r = np.roll(last_values_r, 1)
                # last_values_l[0] = ang_l_deg
                # last_values_r[0] = ang_r_deg

                # avr_l = round(np.average(last_values_l))
                # avr_r = round(np.average(last_values_r))

                cmd = (
                    str(round(-ang_l_deg))
                    + "|"
                    + str(round(ang_r_deg))
                    + "|"
                    + str(round(head_trav_deg))
                    + "\n"
                )
                
                # cmd = (
                #     str(round(0))
                #     + "|"
                #     + str(round(0))
                #     + "|"
                #     + str(round(0))
                #     + "\n"
                # )
                
                await send(cmd.encode())
                # cmd = "b" + str(avr_r) + "\n"
                # send(cmd.encode())

            else:
                print("No human found.")

    print("done.")


def unit_vector(vector):
    """Returns the unit vector of the vector."""
    return vector / np.linalg.norm(vector)


def cos_angle_between(v1, v2):
    """Returns the angle in radians between vectors 'v1' and 'v2'::

    >>> angle_between((1, 0, 0), (0, 1, 0))
    1.5707963267948966
    >>> angle_between((1, 0, 0), (1, 0, 0))
    0.0
    >>> angle_between((1, 0, 0), (-1, 0, 0))
    3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)


# Run the main async program.
if __name__ == "__main__":
    with suppress(asyncio.CancelledError):
        asyncio.run(main())
