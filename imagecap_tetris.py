import cv2
import os
import pygame
import random
import time
from multiprocessing import Process, Array
import mediapipe as mp


# -------- Global Variables -------- #

col = 10  # 10 columns
row = 20  # 20 rows
s_width = 800  # window width
s_height = 750  # window height
play_width = 300  # play window width; 300/10 = 30 width per block
play_height = 600  # play window height; 600/20 = 20 height per block
block_size = 30  # size of block
top_left_x = (s_width - play_width) // 2
top_left_y = s_height - play_height - 50
filepath = "tetris_visuals/highscore.txt"
fontpath = "tetris_visuals/arcade.TTF"
fontpath_mario = "tetris_visuals/mario.ttf"
# shapes formats
S = [
    [".....", ".....", "..00.", ".00..", "....."],
    [".....", "..0..", "..00.", "...0.", "....."],
]

Z = [
    [".....", ".....", ".00..", "..00.", "....."],
    [".....", "..0..", ".00..", ".0...", "....."],
]

I = [
    [".....", "..0..", "..0..", "..0..", "..0.."],
    [".....", "0000.", ".....", ".....", "....."],
]

O = [[".....", ".....", ".00..", ".00..", "....."]]

J = [
    [".....", ".0...", ".000.", ".....", "....."],
    [".....", "..00.", "..0..", "..0..", "....."],
    [".....", ".....", ".000.", "...0.", "....."],
    [".....", "..0..", "..0..", ".00..", "....."],
]

L = [
    [".....", "...0.", ".000.", ".....", "....."],
    [".....", "..0..", "..0..", "..00.", "....."],
    [".....", ".....", ".000.", ".0...", "....."],
    [".....", ".00..", "..0..", "..0..", "....."],
]

T = [
    [".....", "..0..", ".000.", ".....", "....."],
    [".....", "..0..", "..00.", "..0..", "....."],
    [".....", ".....", ".000.", "..0..", "....."],
    [".....", "..0..", ".00..", "..0..", "....."],
]

# index represents the shape
shapes = [S, Z, I, O, J, L, T]
shape_colors = [
    (0, 255, 0),
    (255, 0, 0),
    (0, 255, 255),
    (255, 255, 0),
    (255, 165, 0),
    (0, 0, 255),
    (128, 0, 128),
]

# -------- Image Capture and Multithreading -------- #


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

        # Draw a horizontal blue dotted line at y = 0.3
        line_color = (48, 172, 119)
        line_y = int(0.3 * img.shape[0])
        cv2.line(img, (0, line_y), (img.shape[1], line_y), line_color, 2, cv2.LINE_8, 0)

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


# -------- Tetris -------- #
# class to represent each of the pieces


class Piece(object):
    def __init__(self, x, y, shape):
        self.x = x
        self.y = y
        self.shape = shape
        # choose color from the shape_color list
        self.color = shape_colors[shapes.index(shape)]
        self.rotation = 0  # chooses the rotation according to index


# initialise the grid
def create_grid(locked_pos={}):
    grid = [
        [(0, 0, 0) for x in range(col)] for y in range(row)
    ]  # grid represented rgb tuples

    # locked_positions dictionary
    # (x,y):(r,g,b)
    for y in range(row):
        for x in range(col):
            if (x, y) in locked_pos:
                color = locked_pos[
                    (x, y)
                ]  # get the value color (r,g,b) from the locked_positions dictionary using key (x,y)
                grid[y][x] = color  # set grid position to color

    return grid


def convert_shape_format(piece):
    positions = []
    # get the desired rotated shape from piece
    shape_format = piece.shape[piece.rotation % len(piece.shape)]

    """
    e.g.
       ['.....',
        '.....',
        '..00.',
        '.00..',
        '.....']
    """
    for i, line in enumerate(shape_format):  # i gives index; line gives string
        row = list(line)  # makes a list of char from string
        # j gives index of char; column gives char
        for j, column in enumerate(row):
            if column == "0":
                positions.append((piece.x + j, piece.y + i))

    for i, pos in enumerate(positions):
        # offset according to the input given with dot and zero
        positions[i] = (pos[0] - 2, pos[1] - 4)

    return positions


# checks if current position of piece in grid is valid
def valid_space(piece, grid):
    # makes a 2D list of all the possible (x,y)
    accepted_pos = [
        [(x, y) for x in range(col) if grid[y][x] == (0, 0, 0)] for y in range(row)
    ]
    # removes sub lists and puts (x,y) in one list; easier to search
    accepted_pos = [x for item in accepted_pos for x in item]

    formatted_shape = convert_shape_format(piece)

    for pos in formatted_shape:
        if pos not in accepted_pos:
            if pos[1] >= 0:
                return False
    return True


# check if piece is out of board
def check_lost(positions):
    for pos in positions:
        x, y = pos
        if y < 1:
            return True
    return False


# chooses a shape randomly from shapes list
def get_shape():
    return Piece(5, 0, random.choice(shapes))


# draws text in the middle
def draw_text_middle(text, size, color, surface):
    font = pygame.font.Font(fontpath, size)  # , bold=False, italic=True)
    label = font.render(text, 1, color)

    surface.blit(
        label,
        (
            top_left_x + play_width / 2 - (label.get_width() / 2),
            top_left_y + play_height / 2 - (label.get_height() / 2),
        ),
    )


# draws the lines of the grid for the game
def draw_grid(surface):
    r = g = b = 0
    grid_color = (r, g, b)

    for i in range(row):
        # draw grey horizontal lines
        pygame.draw.line(
            surface,
            grid_color,
            (top_left_x, top_left_y + i * block_size),
            (top_left_x + play_width, top_left_y + i * block_size),
        )
        for j in range(col):
            # draw grey vertical lines
            pygame.draw.line(
                surface,
                grid_color,
                (top_left_x + j * block_size, top_left_y),
                (top_left_x + j * block_size, top_left_y + play_height),
            )


# clear a row when it is filled
def clear_rows(grid, locked):
    # need to check if row is clear then shift every other row above down one
    increment = 0
    for i in range(len(grid) - 1, -1, -1):  # start checking the grid backwards
        grid_row = grid[i]  # get the last row
        if (
            0,
            0,
            0,
        ) not in grid_row:  # if there are no empty spaces (i.e. black blocks)
            increment += 1
            # add positions to remove from locked
            index = i  # row index will be constant
            for j in range(len(grid_row)):
                try:
                    # delete every locked element in the bottom row
                    del locked[(j, i)]
                except ValueError:
                    continue

    # shift every row one step down
    # delete filled bottom row
    # add another empty row on the top
    # move down one step
    if increment > 0:
        # sort the locked list according to y value in (x,y) and then reverse
        # reversed because otherwise the ones on the top will overwrite the lower ones
        for key in sorted(list(locked), key=lambda a: a[1])[::-1]:
            x, y = key
            if y < index:  # if the y value is above the removed index
                new_key = (x, y + increment)  # shift position to down
                locked[new_key] = locked.pop(key)

    return increment


# draws the upcoming piece
def draw_next_shape(piece, surface):
    font = pygame.font.Font(fontpath, 30)
    label = font.render("Next shape", 1, (255, 255, 255))

    start_x = top_left_x + play_width + 50
    start_y = top_left_y + (play_height / 2 - 100)

    shape_format = piece.shape[piece.rotation % len(piece.shape)]

    for i, line in enumerate(shape_format):
        row = list(line)
        for j, column in enumerate(row):
            if column == "0":
                pygame.draw.rect(
                    surface,
                    piece.color,
                    (
                        start_x + j * block_size,
                        start_y + i * block_size,
                        block_size,
                        block_size,
                    ),
                    0,
                )

    surface.blit(label, (start_x, start_y - 30))

    # pygame.display.update()


# draws the content of the window
def draw_window(surface, grid, score=0, last_score=0):
    surface.fill((0, 0, 0))  # fill the surface with black

    pygame.font.init()  # initialise font
    font = pygame.font.Font(fontpath_mario, 65)  # , bold=True)
    # initialise 'Tetris' text with white
    label = font.render("TETRIS", 1, (255, 255, 255))

    # put surface on the center of the window
    surface.blit(label, ((top_left_x + play_width / 2) - (label.get_width() / 2), 30))

    # current score
    font = pygame.font.Font(fontpath, 30)
    label = font.render("SCORE   " + str(score), 1, (255, 255, 255))

    start_x = top_left_x + play_width + 50
    start_y = top_left_y + (play_height / 2 - 100)

    surface.blit(label, (start_x, start_y + 200))

    # last score
    label_hi = font.render("HIGHSCORE   " + str(last_score), 1, (255, 255, 255))

    start_x_hi = top_left_x - 240
    start_y_hi = top_left_y + 200

    surface.blit(label_hi, (start_x_hi + 20, start_y_hi + 200))

    # draw content of the grid
    for i in range(row):
        for j in range(col):
            # pygame.draw.rect()
            # draw a rectangle shape
            # rect(Surface, color, Rect, width=0) -> Rect
            pygame.draw.rect(
                surface,
                grid[i][j],
                (
                    top_left_x + j * block_size,
                    top_left_y + i * block_size,
                    block_size,
                    block_size,
                ),
                0,
            )

    # draw vertical and horizontal grid lines
    draw_grid(surface)

    # draw rectangular border around play area
    border_color = (255, 255, 255)
    pygame.draw.rect(
        surface, border_color, (top_left_x, top_left_y, play_width, play_height), 4
    )

    # pygame.display.update()


# update the score txt file with high score
def update_score(new_score):
    score = get_max_score()

    with open(filepath, "w") as file:
        if new_score > score:
            file.write(str(new_score))
        else:
            file.write(str(score))


# get the high score from the file
def get_max_score():
    with open(filepath, "r") as file:
        lines = file.readlines()  # reads all the lines and puts in a list
        score = int(lines[0].strip())  # remove \n

    return score


def main(window, shared_array):
    locked_positions = {}
    create_grid(locked_positions)

    change_piece = False
    run = True
    current_piece = get_shape()
    next_piece = get_shape()
    clock = pygame.time.Clock()
    fall_time = 0
    fall_speed = 0.35
    level_time = 0
    score = 0
    last_score = get_max_score()

    cooldown_duration = 0.5
    cooldown_timer = CustomTimer(cooldown_duration)
    while run:
        # need to constantly make new grid as locked positions always change
        grid = create_grid(locked_positions)

        # helps run the same on every computer
        # add time since last tick() to fall_time
        fall_time += clock.get_rawtime()  # returns in milliseconds
        level_time += clock.get_rawtime()

        clock.tick()  # updates clock

        if level_time / 1000 > 5:  # make the difficulty harder every 10 seconds
            level_time = 0
            if fall_speed > 0.15:  # until fall speed is 0.15
                fall_speed -= 0.005

        if fall_time / 1000 > fall_speed:
            fall_time = 0
            current_piece.y += 1
            if not valid_space(current_piece, grid) and current_piece.y > 0:
                current_piece.y -= 1
                # since only checking for down - either reached bottom or hit another piece
                # need to lock the piece position
                # need to generate new piece
                change_piece = True

        l_wrist_y = shared_array[1]
        r_wrist_y = shared_array[5]
        y_threshold = 0.3

        if cooldown_timer.is_active():
            pass

        # Rotate
        elif l_wrist_y < y_threshold and r_wrist_y < y_threshold:
            print(
                f"Rotating: Both wrists above threshold (L = {l_wrist_y}, R = {r_wrist_y})"
            )
            current_piece.rotation = current_piece.rotation + 1 % len(
                current_piece.shape
            )
            if not valid_space(current_piece, grid):
                current_piece.rotation = current_piece.rotation - 1 % len(
                    current_piece.shape
                )
            cooldown_timer.reset()
        # Move left
        elif l_wrist_y < y_threshold:
            print(
                f"Moving left: Left wrist above threshold, right below (L = {l_wrist_y}, R = {r_wrist_y})"
            )
            current_piece.x -= 1  # move x position left
            if not valid_space(current_piece, grid):
                current_piece.x += 1
            cooldown_timer.reset()

        # Move right
        elif r_wrist_y < y_threshold:
            print(
                f"Moving right: Right wrist above threshold, left below (L = {l_wrist_y}, R = {r_wrist_y})"
            )
            current_piece.x += 1  # move x position right
            if not valid_space(current_piece, grid):
                current_piece.x -= 1
            cooldown_timer.reset()

        piece_pos = convert_shape_format(current_piece)

        # draw the piece on the grid by giving color in the piece locations
        for i in range(len(piece_pos)):
            x, y = piece_pos[i]
            if y >= 0:
                grid[y][x] = current_piece.color

        if change_piece:  # if the piece is locked
            for pos in piece_pos:
                p = (pos[0], pos[1])
                # add the key and value in the dictionary
                locked_positions[p] = current_piece.color
            current_piece = next_piece
            next_piece = get_shape()
            change_piece = False
            # increment score by 10 for every row cleared
            score += clear_rows(grid, locked_positions) * 10
            update_score(score)

            if last_score < score:
                last_score = score

        draw_window(window, grid, score, last_score)
        draw_next_shape(next_piece, window)
        pygame.display.update()

        if check_lost(locked_positions):
            run = False

    draw_text_middle("You Lost", 40, (255, 255, 255), window)
    pygame.display.update()
    pygame.time.delay(2000)  # wait for 2 seconds
    pygame.quit()


def main_menu(shared_array):
    os.environ["SDL_VIDEO_WINDOW_POS"] = "%d,%d" % (500, 0)
    pygame.font.init()
    win = pygame.display.set_mode((s_width, s_height))
    pygame.display.set_caption("Tetris")
    run = True
    while run:
        opening = "Raise  both  hands  to  begin"
        draw_text_middle(opening, 50, (255, 255, 255), win)
        pygame.display.update()

        l_wrist_y = shared_array[1]
        r_wrist_y = shared_array[5]
        y_threshold = 0.3

        if l_wrist_y < y_threshold and r_wrist_y < y_threshold:
            main(win, shared_array)

    pygame.quit()


# -------- Main -------- #


if __name__ == "__main__":
    shared_array = Array("d", [0.0] * 8)

    process1 = Process(target=image_capture, args=(shared_array,))
    process2 = Process(target=main_menu, args=(shared_array,))

    print("Starting Image Capture...")
    process1.start()
    time.sleep(5)
    print("Starting Tetris...")
    process2.start()

    process2.join()
    process1.join()
