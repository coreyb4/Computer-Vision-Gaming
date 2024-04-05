import cv2
import os
import pygame as pg
import random
import sys
import time
from multiprocessing import Process, Array
import mediapipe as mp
from mario_data import main
from mario_data import menu
from mario_data import config as c


# -------- Image Capture and Multithreading -------- #


def image_capture(shared_array):
    # mp and cv2 init.
    mpPose = mp.solutions.pose
    pose = mpPose.Pose()
    mpDraw = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(1)
    pTime = 0

    # Set window size
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Image", 800, 750)  # Adjust dimensions as needed

    # Landmarks to display
    landmark_indices = [
        mpPose.PoseLandmark.LEFT_SHOULDER,
        mpPose.PoseLandmark.RIGHT_SHOULDER,
        mpPose.PoseLandmark.LEFT_ELBOW,
        mpPose.PoseLandmark.RIGHT_ELBOW,
        mpPose.PoseLandmark.LEFT_WRIST,
        mpPose.PoseLandmark.RIGHT_WRIST,
        mpPose.PoseLandmark.LEFT_HIP,
        mpPose.PoseLandmark.RIGHT_HIP,
        mpPose.PoseLandmark.LEFT_KNEE,
        mpPose.PoseLandmark.RIGHT_KNEE,
        mpPose.PoseLandmark.LEFT_ANKLE,
        mpPose.PoseLandmark.RIGHT_ANKLE,
        mpPose.PoseLandmark.NOSE,
    ]

    # Connections to display
    connections = [
        (mpPose.PoseLandmark.LEFT_SHOULDER, mpPose.PoseLandmark.RIGHT_SHOULDER),
        (mpPose.PoseLandmark.RIGHT_ELBOW, mpPose.PoseLandmark.RIGHT_SHOULDER),
        (mpPose.PoseLandmark.LEFT_ELBOW, mpPose.PoseLandmark.LEFT_SHOULDER),
        (mpPose.PoseLandmark.LEFT_WRIST, mpPose.PoseLandmark.LEFT_ELBOW),
        (mpPose.PoseLandmark.RIGHT_WRIST, mpPose.PoseLandmark.RIGHT_ELBOW),
        (mpPose.PoseLandmark.LEFT_HIP, mpPose.PoseLandmark.LEFT_SHOULDER),
        (mpPose.PoseLandmark.RIGHT_HIP, mpPose.PoseLandmark.RIGHT_SHOULDER),
        (mpPose.PoseLandmark.LEFT_KNEE, mpPose.PoseLandmark.LEFT_HIP),
        (mpPose.PoseLandmark.RIGHT_KNEE, mpPose.PoseLandmark.RIGHT_HIP),
        (mpPose.PoseLandmark.LEFT_ANKLE, mpPose.PoseLandmark.LEFT_KNEE),
        (mpPose.PoseLandmark.RIGHT_ANKLE, mpPose.PoseLandmark.RIGHT_KNEE),
        (mpPose.PoseLandmark.RIGHT_HIP, mpPose.PoseLandmark.LEFT_HIP),
    ]

    # Video capture loop
    while True:
        # Capture and process image
        success, img = cap.read()
        img = cv2.flip(img, 1)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(imgRGB)

        if results.pose_landmarks:
            # Extract desired landmarks (flip L & R bc image is mirrored)
            l_wrist = results.pose_landmarks.landmark[mpPose.PoseLandmark.RIGHT_WRIST]
            r_wrist = results.pose_landmarks.landmark[mpPose.PoseLandmark.LEFT_WRIST]
            nose = results.pose_landmarks.landmark[mpPose.PoseLandmark.NOSE]

            # Update shared array
            shared_array[:4] = [l_wrist.x, l_wrist.y, l_wrist.z, l_wrist.visibility]
            shared_array[4:] = [r_wrist.x, r_wrist.y, nose.x, nose.y]

            # Place points on image
            for i in landmark_indices:
                lm = results.pose_landmarks.landmark[i]
                if lm.visibility > 0.5:
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

            # Draw lines between selected landmarks
            for connection in connections:
                start_idx, end_idx = connection
                start_lm = results.pose_landmarks.landmark[start_idx]
                end_lm = results.pose_landmarks.landmark[end_idx]
                if start_lm.visibility > 0.5 and end_lm.visibility > 0.5:
                    start_point = (int(start_lm.x * w), int(start_lm.y * h))
                    end_point = (int(end_lm.x * w), int(end_lm.y * h))
                    cv2.line(
                        img, start_point, end_point, (255, 255, 255), 2, cv2.LINE_AA
                    )
        line_color = (48, 172, 119)
        # # Draw a horizontal green line at y = 0.3

        # line_y = int(0.3 * img.shape[0])
        # cv2.line(img, (0, line_y), (img.shape[1], line_y), line_color, 2, cv2.LINE_8, 0)

        # Line 1: x = 0 to x = 0.3, y = 0.3
        line1_start_x = 0
        line1_end_x = int(0.3 * img.shape[1])
        line1_y = int(0.3 * img.shape[0])
        cv2.line(
            img,
            (line1_start_x, line1_y),
            (line1_end_x, line1_y),
            line_color,
            2,
            cv2.LINE_8,
            0,
        )

        # Line 2: x = 0.7 to x = 1, y = 0.3
        line2_start_x = int(0.7 * img.shape[1])
        line2_end_x = img.shape[1]
        line2_y = int(0.3 * img.shape[0])
        cv2.line(
            img,
            (line2_start_x, line2_y),
            (line2_end_x, line2_y),
            line_color,
            2,
            cv2.LINE_8,
            0,
        )

        # Line 3: x = 0.3 to x = 0.7, y = 0.15
        line3_start_x = int(0.3 * img.shape[1])
        line3_end_x = int(0.7 * img.shape[1])
        line3_y = int(0.15 * img.shape[0])
        cv2.line(
            img,
            (line3_start_x, line3_y),
            (line3_end_x, line3_y),
            line_color,
            2,
            cv2.LINE_8,
            0,
        )
        line_y = int(0.3 * img.shape[0])
        # Draw a vertical line at x = 0.3 from y = 0 to y = 0.3
        line_color = (48, 172, 119)  # Red color
        line_x1 = int(0.3 * img.shape[1])
        cv2.line(img, (line_x1, 0), (line_x1, line_y), line_color, 2, cv2.LINE_8, 0)

        # Draw a vertical line at x = 0.7 from y = 0 to y = 0.3
        line_color = (48, 172, 119)  # Red color
        line_x2 = int(0.7 * img.shape[1])
        cv2.line(img, (line_x2, 0), (line_x2, line_y), line_color, 2, cv2.LINE_8, 0)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


# -------- Cooldown timer -------- #
class CustomTimer:
    def __init__(self, interval):
        self.interval = interval
        self.start_time = time.time()
        self.active = True

    def is_active(self):
        current_time = time.time()
        if current_time - self.start_time >= self.interval:
            self.active = False
        return self.active

    def reset(self):
        self.start_time = time.time()
        self.active = True


class App:
    def __init__(self):
        self.menu = None
        self.main = None

    def run(self, shared_array):
        self.menu = menu.Menu()
        self.menu.menu_loop()
        if (
            self.menu.quit_state == "play"
        ):  # Check whether to continue to game or quit app
            self.main = main.Main()
            self.main.main_loop(shared_array)
            if self.main.quit_state == "menu":
                # If you think this is a cheat
                # to avoid destroying instances,
                # you are right, I'm just too
                # lazy to do that.
                os.execl(sys.executable, sys.executable, *sys.argv)  # Restart game


def mario_gameloop(shared_array):
    os.environ["SDL_VIDEO_WINDOW_POS"] = "%d,%d" % (1000, -930)
    pg.init()  # Initialize pygame module
    c.screen = pg.display.set_mode((c.SCREEN_SIZE.x, c.SCREEN_SIZE.y))
    pg.display.set_caption(c.CAPTION)
    c.clock = pg.time.Clock()

    app = App()
    app.run(shared_array)

    pg.quit()


if __name__ == "__main__":
    shared_array = Array("d", [0.0] * 8)

    process1 = Process(target=image_capture, args=(shared_array,))
    process2 = Process(target=mario_gameloop, args=(shared_array,))

    print("Starting Image Capture...")
    process1.start()
    time.sleep(3)
    print("Starting Mario Bros...")
    process2.start()

    process2.join()
    process1.join()
