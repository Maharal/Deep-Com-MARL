# Standard package import
from abc import ABCMeta
from random import randint, choice, random
from string import ascii_uppercase, digits

# Local application import
from PyGameFacade import *
from Interfaces import IEnvironment


class ObjectBase(metaclass = ABCMeta):
    def __init__(self, diam : float, color : tuple, constrain : tuple, stroke_weight : int = 0):
        self.constrain = constrain
        self.diam = diam
        self.radius = 0.5 * diam
        self.color = color
        self.stroke_weight = stroke_weight
        self.frozen = False
        self.new_pos()

    def new_pos(self):
        x_min, y_min, x_max, y_max = self.constrain
        self.pos = Vector2(randint(x_min, x_max), randint(y_min, y_max))

    def draw(self, screen : Surface):
        dimensions = (self.pos.x - self.radius, self.pos.y - self.radius, self.diam, self.diam)
        ellipse(screen, CYAN if self.frozen else self.color, dimensions, self.stroke_weight)

    def has_intersection(self, obj : 'ObjectBase') -> bool:
        return self.radius + obj.radius >= (obj.pos - self.pos).magnitude()

    def is_inside(self, obj : 'ObjectBase') -> bool:
        return obj.radius >= self.radius + (obj.pos - self.pos).magnitude()

class MovableBase(ObjectBase):
    def __init__(self, diam : float, color : tuple, constrain : tuple, stroke_weight : int = 0):
        super().__init__(diam, color, constrain, stroke_weight)
        self.vel = Vector2()
        self.acc = Vector2()

    def add_force(self, force):
        self.acc += force

    def next_state(self):
        self.pos += self.vel * dt
        self.acc = Vector2(0, 0)