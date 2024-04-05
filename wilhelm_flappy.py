import cv2
import os
import pygame
import random
import sys
import time
from multiprocessing import Process, Array
import mediapipe as mp


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

            # Update shared array
            shared_array[:4] = [l_wrist.x, l_wrist.y, l_wrist.z, l_wrist.visibility]
            shared_array[4:] = [r_wrist.x, r_wrist.y, r_wrist.z, r_wrist.visibility]

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

        # Draw a horizontal green line at y = 0.3
        line_color = (48, 172, 119)
        line_y = int(0.3 * img.shape[0])
        cv2.line(img, (0, line_y), (img.shape[1], line_y), line_color, 2, cv2.LINE_8, 0)

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


os.environ["SDL_VIDEO_WINDOW_POS"] = "%d,%d" % (1100, -1060)
pygame.mixer.pre_init(frequency=44100, size=16, channels=1, buffer=512)
pygame.init()
screen = pygame.display.set_mode((576, 1024))
clock = pygame.time.Clock()
game_font = pygame.font.Font("flappybird_assets/04B_19.ttf", 40)

# Game Variables
gravity = 0.15
bird_movement = 0
game_active = True
score = 0
high_score = 0

bg_surface = pygame.image.load("flappybird_assets/background-day.png").convert()
bg_surface = pygame.transform.scale2x(bg_surface)

floor_surface = pygame.image.load("flappybird_assets/base.png").convert()
floor_surface = pygame.transform.scale2x(floor_surface)
floor_x_pos = 0

bird_downflap = pygame.transform.scale2x(
    pygame.image.load("flappybird_assets/yellowbird-downflap.png").convert_alpha()
)
bird_midflap = pygame.transform.scale2x(
    pygame.image.load("flappybird_assets/yellowbird-midflap.png").convert_alpha()
)
bird_upflap = pygame.transform.scale2x(
    pygame.image.load("flappybird_assets/yellowbird-upflap.png").convert_alpha()
)
bird_frames = [bird_downflap, bird_midflap, bird_upflap]
bird_index = 0
bird_surface = bird_frames[bird_index]
bird_rect = bird_surface.get_rect(center=(100, 512))

BIRDFLAP = pygame.USEREVENT + 1
pygame.time.set_timer(BIRDFLAP, 200)

# bird_surface = pygame.image.load('assets/bluebird-midflap.png').convert_alpha()
# bird_surface = pygame.transform.scale2x(bird_surface)
# bird_rect = bird_surface.get_rect(center = (100,512))

pipe_surface = pygame.image.load("flappybird_assets/pipe-green.png")
pipe_surface = pygame.transform.scale2x(pipe_surface)
pipe_list = []
SPAWNPIPE = pygame.USEREVENT
pygame.time.set_timer(SPAWNPIPE, 4000)
pipe_height = [400, 600, 800]

game_over_surface = pygame.transform.scale2x(
    pygame.image.load("flappybird_assets/message.png").convert_alpha()
)
game_over_rect = game_over_surface.get_rect(center=(288, 512))

flap_sound = pygame.mixer.Sound("flappybird_sound/sfx_wing.wav")
death_sound = pygame.mixer.Sound(
    "flappybird_sound\VOXScrm_Wilhelm scream (ID 0477)_BSB.wav"
)
score_sound = pygame.mixer.Sound("flappybird_sound/sfx_point.wav")
score_sound_countdown = 100


def draw_floor():
    screen.blit(floor_surface, (floor_x_pos, 900))
    screen.blit(floor_surface, (floor_x_pos + 576, 900))


def create_pipe():
    random_pipe_pos = random.choice(pipe_height)
    bottom_pipe = pipe_surface.get_rect(midtop=(800, random_pipe_pos))
    top_pipe = pipe_surface.get_rect(midbottom=(800, random_pipe_pos - 300))
    return bottom_pipe, top_pipe


def move_pipes(pipes):
    for pipe in pipes:
        pipe.centerx -= 3
    return pipes


def draw_pipes(pipes):
    for pipe in pipes:
        if pipe.bottom >= 1024:
            screen.blit(pipe_surface, pipe)
        else:
            flip_pipe = pygame.transform.flip(pipe_surface, False, True)
            screen.blit(flip_pipe, pipe)


def remove_pipes(pipes):
    for pipe in pipes:
        if pipe.centerx == -600:
            pipes.remove(pipe)
    return pipes


def check_collision(pipes):
    for pipe in pipes:
        if bird_rect.colliderect(pipe):
            death_sound.play()
            return False

    if bird_rect.top <= -100 or bird_rect.bottom >= 900:
        return False

    return True


def rotate_bird(bird):
    new_bird = pygame.transform.rotozoom(bird, -bird_movement * 3, 1)
    return new_bird


def bird_animation():
    new_bird = bird_frames[bird_index]
    new_bird_rect = new_bird.get_rect(center=(100, bird_rect.centery))
    return new_bird, new_bird_rect


def score_display(game_state):
    if game_state == "main_game":
        score_surface = game_font.render(str(int(score)), True, (255, 255, 255))
        score_rect = score_surface.get_rect(center=(288, 100))
        screen.blit(score_surface, score_rect)
    if game_state == "game_over":
        score_surface = game_font.render(f"Score: {int(score)}", True, (255, 255, 255))
        score_rect = score_surface.get_rect(center=(288, 100))
        screen.blit(score_surface, score_rect)

        high_score_surface = game_font.render(
            f"High score: {int(high_score)}", True, (255, 255, 255)
        )
        high_score_rect = high_score_surface.get_rect(center=(288, 850))
        screen.blit(high_score_surface, high_score_rect)


def update_score(score, high_score):
    if score > high_score:
        high_score = score
    return high_score


def flappy_gameloop(shared_array):
    global game_active, bird_movement, bird_surface, bird_rect, pipe_list, score, score_sound_countdown, floor_x_pos, bird_index, high_score

    cooldown_duration = 0.2
    cooldown_timer = CustomTimer(cooldown_duration)

    scored = False
    while True:
        scored = False
        l_wrist_y = shared_array[1]
        r_wrist_y = shared_array[5]
        y_threshold = 0.3

        for event in pygame.event.get():
            if not cooldown_timer.is_active() and (
                l_wrist_y < y_threshold or r_wrist_y < y_threshold
            ):
                if game_active:
                    bird_movement = 0
                    bird_movement -= 8
                    flap_sound.play()
                if game_active == False:
                    game_active = True
                    pipe_list.clear()
                    bird_rect.center = (100, 512)
                    bird_movement = 0
                    score = 0
                cooldown_timer.reset()

            if event.type == SPAWNPIPE:
                pipe_list.extend(create_pipe())

            if event.type == BIRDFLAP:
                if bird_index < 2:
                    bird_index += 1
                else:
                    bird_index = 0

                bird_surface, bird_rect = bird_animation()

        screen.blit(bg_surface, (0, 0))

        if game_active:
            # Bird
            bird_movement += gravity
            rotated_bird = rotate_bird(bird_surface)
            bird_rect.centery += bird_movement
            screen.blit(rotated_bird, bird_rect)
            game_active = check_collision(pipe_list)

            # Pipes
            pipe_list = move_pipes(pipe_list)
            pipe_list = remove_pipes(pipe_list)
            draw_pipes(pipe_list)

            score += 0.01 / 3
            score_display("main_game")
        else:
            screen.blit(game_over_surface, game_over_rect)
            high_score = update_score(score, high_score)
            score_display("game_over")

        # Floor
        floor_x_pos -= 1
        draw_floor()
        if floor_x_pos <= -576:
            floor_x_pos = 0

        pygame.display.update()
        clock.tick(80)


if __name__ == "__main__":
    shared_array = Array("d", [0.0] * 8)

    process1 = Process(target=image_capture, args=(shared_array,))
    process2 = Process(target=flappy_gameloop, args=(shared_array,))

    print("Starting Image Capture...")
    process1.start()
    time.sleep(5)
    print("Starting Flappy Bird...")
    process2.start()

    process2.join()
    process1.join()
