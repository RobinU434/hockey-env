import math


FPS = 50
SCALE = 60.0  # affects how fast-paced the game is, forces should be adjusted as well (Don't touch)

VIEWPORT_W = 600
VIEWPORT_H = 480
W = VIEWPORT_W / SCALE
H = VIEWPORT_H / SCALE
CENTER_X = W / 2
CENTER_Y = H / 2
ZONE = W / 20
MAX_ANGLE = math.pi / 3  # Maximimal angle of racket
MAX_TIME_KEEP_PUCK = 15
GOAL_SIZE = 75

RACKETPOLY = [(-10, 20), (+5, 20), (+5, -20), (-10, -20), (-18, -10), (-21, 0), (-18, 10)]
RACKETFACTOR = 1.2

FORCEMULTIPLIER = 6000
SHOOTFORCEMULTIPLIER = 60
TORQUEMULTIPLIER = 400
MAX_PUCK_SPEED = 25
