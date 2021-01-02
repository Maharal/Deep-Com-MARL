# Standard import
from os import environ
environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
environ['SDL_AUDIODRIVER'] = 'dsp'

# Third part package import
from typing import Callable, List
from pygame.draw import rect, ellipse
from pygame import init, quit
from pygame.display import set_mode as set_size, update as update_screen
from pygame.event import get as get_event
from pygame.constants import QUIT
from pygame.math import Vector2
from pygame import Surface
from pygame import K_RETURN, KEYDOWN


global dt


# Colors
WHITE = (255, 255, 255)
GRAY  = (128, 128, 128)
BLACK = (0, 0, 0)
RED   = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE  = (0, 0, 255)
CYAN  = (0, 255, 255)
dt = 0.5


# Complement functions 
def background(screen : Surface, color : tuple):
    screen.fill(color)


def constrain(value, min_value, max_value):
    if value < min_value:
        return min_value
    elif value > max_value:
        return max_value
    else:
        return value