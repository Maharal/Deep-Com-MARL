# standard libraries import
from random import random

# Third part import
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.random import normal

# Local application import
from PyGameFacade import *
from Interfaces import IEnvironment
from Base import ObjectBase, MovableBase


class Brain(nn.Module):
    def __init__(self, n_landmark : int, n_agents : int, size_channel : int, hidden_size = 16):
        super(Brain, self).__init__()
        self.lstm1 = nn.LSTM(2 * 2 * n_landmark + (n_agents - 1) * size_channel + 4, hidden_size, batch_first = True)
        self.seq = nn.Sequential(nn.Linear(hidden_size, 32), nn.ReLU(), nn.Linear(32, 2 + size_channel), nn.Sigmoid())
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.h = torch.zeros(1, 1, hidden_size)
        self.c = torch.zeros(1, 1, hidden_size)
        
    def forward(self, x):
        x = x.reshape(1, 1, -1)
        _, (self.h, self.c) = self.lstm1(x, (self.h, self.c))
        x = self.seq(self.h)
        x = x.reshape(-1)
        return x


class Agent(MovableBase):
    def __init__(self, agent_param : dict, n_agents : int, n_landmarks : int, size_chanel : int, size_epoch : int, vel_const : float = 10):
        super().__init__(**agent_param)
        self.brain = Brain(n_landmarks, n_agents, size_chanel)
        self.size_chanel = size_chanel
        self.size_epoch = size_epoch
        self.epoch = 0
        self.reward_memory = []
        self.actions_memory = []
        self.mensage = torch.zeros(size_chanel)
        self.vel_const = vel_const
        self.forward = True
        self.vel = Vector2(vel_const * normal(), vel_const * normal())

    def __perception(self, environment):
        msg = [agent.mensage for agent in environment.agents if agent is not self]
        position = [torch.Tensor([self.pos.x, self.pos.y])]
        velocity = [torch.Tensor([self.vel.x, self.vel.y])]
        landmarks = [torch.Tensor([landmark.pos.x, landmark.pos.y]) for landmark in environment.landmarks]
        goals = [torch.Tensor([goal.pos.x, goal.pos.y]) for goal in environment.goals]
        return torch.cat(msg + position + velocity + landmarks + goals)

    def __walk(self, teta : float, r : float):
        self.vel.rotate_ip(45 if random() <= teta else 0)
        self.vel.rotate_ip(-45 if random() <= teta else 0)
        self.forward = random() <= r

    def __forward(self):
        self.vel += dt * self.acc;
        self.vel.scale_to_length(self.vel_const) 
        if self.forward:
            self.pos += dt * self.vel
            self.forward = False
        self.acc = Vector2()

    def next_state(self, environment : IEnvironment):
        event = self.__perception(environment)
        actions = self.brain.forward(event)
        self.__walk(actions[0], actions[1])  
        self.mensage = actions[2:]
        self.__forward()


class Landmark(MovableBase):
    def __init__(self, landmark_param : dict, thresold : float = 5):
        super().__init__(**landmark_param)
        self.thresold = thresold
        self.count = 0

    def next_state(self):
        self.count += 1
        if self.thresold < self.acc.magnitude():
            self.vel += self.acc * dt
            print(self.count)
        super().next_state()


class Goal(ObjectBase):
    def __init__(self, goal_param : dict): 
        super().__init__(**goal_param)