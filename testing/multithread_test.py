import cv2
import time
from multiprocessing import Process, Array
import mediapipe as mp


def image_capture(shared_array):
    # mp and cv2 init.
    mpPose = mp.solutions.pose
    pose = mpPose.Pose()
    mpDraw = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(1)
    pTime = 0

    # Video capture loop
    while True:
        # Capture and process image
        success, img = cap.read()
        img = cv2.flip(img, 1)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(imgRGB)

        if results.pose_landmarks:
            # Place landmarks and connections on display window
            mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

            # Extract desired landmarks (flip L & R bc image is mirrored)
            l_wrist = results.pose_landmarks.landmark[mpPose.PoseLandmark.RIGHT_WRIST]
            r_wrist = results.pose_landmarks.landmark[mpPose.PoseLandmark.LEFT_WRIST]

            # Update shared array
            shared_array[:4] = [l_wrist.x, l_wrist.y, l_wrist.z, l_wrist.visibility]
            shared_array[4:] = [r_wrist.x, r_wrist.y, r_wrist.z, r_wrist.visibility]

            # Place better-looking points on image
            for lm in results.pose_landmarks.landmark:
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


def script2(shared_array):
    while True:
        # Script 2 logic
        l_wrist_x, l_wrist_y, l_wrist_z, l_wrist_visibility = shared_array[:4]
        r_wrist_x, r_wrist_y, r_wrist_z, r_wrist_visibility = shared_array[4:]
        if all(val != 0.0 for val in shared_array):
            print(
                "Left wrist details:",
                l_wrist_x,
                l_wrist_y,
                l_wrist_z,
                l_wrist_visibility,
            )
            print(
                "Right wrist details:",
                r_wrist_x,
                r_wrist_y,
                r_wrist_z,
                r_wrist_visibility,
            )
        time.sleep(0.25)


if __name__ == "__main__":
    shared_array = Array("d", [0.0] * 8)

    process1 = Process(target=image_capture, args=(shared_array,))
    process2 = Process(target=script2, args=(shared_array,))

    process2.start()
    process1.start()

    process2.join()
    process1.join()
