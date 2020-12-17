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
        self.seq = nn.Sequential(nn.Linear(hidden_size, 32), nn.ReLU(), nn.Linear(32, 3 + size_channel), nn.Sigmoid())
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.h = torch.zeros(1, 1, hidden_size).to(self.device)
        self.c = torch.zeros(1, 1, hidden_size).to(self.device)
        
    def forward(self, x, batch_size = 1, num_steps = 1):
        x = x.reshape(batch_size, num_steps, -1).to(self.device)
        _, (h, c) = self.lstm1(x, (self.h, self.c))
        self.h = h.detach()
        self.c = c.detach()
        x = self.seq(self.h)
        if batch_size == 1 and num_steps == 1:
            return x.reshape(-1)
        else:
            return x



class Agent(MovableBase):
    def __init__(self, _id : int, agent_param : dict, n_agents : int, n_landmarks : int, size_chanel : int, size_epoch : int, vel_const : float = 10):
        super().__init__(**agent_param)
        self.brain = Brain(n_landmarks, n_agents, size_chanel)
        self.size_chanel = size_chanel
        self.size_epoch = size_epoch
        self.epoch = 0
        self.rewards = []
        self.actions = []
        self.states = []
        self.mensage = torch.zeros(size_chanel)
        self.vel_const = vel_const
        self.forward = False
        self.vel = Vector2(vel_const * normal(), vel_const * normal())
        self._id = _id
        self.reward = 0
        self.has_learn = False
                
    def __perception(self, environment):
        msg = [agent.mensage for agent in environment.agents if agent is not self]
        position = [torch.Tensor([self.pos.x, self.pos.y])]
        velocity = [torch.Tensor([self.vel.x, self.vel.y])]
        landmarks = [torch.Tensor([landmark.pos.x, landmark.pos.y]) for landmark in environment.landmarks]
        goals = [torch.Tensor([goal.pos.x, goal.pos.y]) for goal in environment.goals]
        return torch.cat(msg + position + velocity + landmarks + goals)

    def __walk(self, teta1 : float, teta2 : float, r : float):
        self.vel.rotate_ip(15 * (random() <= teta1))
        self.vel.rotate_ip(-15 * (random() <= teta2))
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
        self.__walk(actions[0], actions[1], actions[2])  
        self.mensage = actions[3:]
        self.__forward()
        self.__update_rewards()
        return actions

    def store(self, file_path : str):
        torch.save(self.brain.state_dict(), file_path)
        
    def load(self, file_path : str):
        self.brain.load_state_dict(torch.load(file_path))
        self.brain.eval()

    def add_reward(self, r):
        self.has_learn = True
        self.reward += r
        if r > 0:
            print(f"R: {r}")
            
    def clear_memory(self):
        self.rewards = []
        self.actions = []
        self.states = []
        self.has_learn = False

    def __update_rewards(self):
        self.rewards.append(self.reward)
        self.reward = 0



class Landmark(MovableBase):
    def __init__(self, landmark_param : dict, thresold : float = 5):
        super().__init__(**landmark_param)
        self.thresold = thresold
        self.monitored_agents = []
        self.frozen = False

    def next_state(self):
        if self.thresold < self.acc.magnitude() and not self.frozen: 
            self.vel += self.acc * dt
            self.reward_monitored(1 * self.vel.magnitude())
        super().next_state()
        
    def monitoring(self, agent : Agent):
        if agent not in self.monitored_agents:
            self.monitored_agents.append(agent)
        
    def reward_monitored(self, reward):
        if len(self.monitored_agents) > 1:
            for agent in self.monitored_agents:
                agent.add_reward(reward)
    
    def clear_monitored(self):
        if self.acc.magnitude() <= 0.01:
            self.monitored_agents = []
        
        
        
class Goal(ObjectBase):
    def __init__(self, goal_param : dict): 
        super().__init__(**goal_param)