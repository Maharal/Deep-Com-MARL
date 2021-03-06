# standard libraries import
from random import random

# Third part import
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.random import normal
import numpy as np
from time import sleep
import matplotlib.pyplot as plt
from torch.distributions.bernoulli import Bernoulli

# Local application import
from PyGameFacade import *
from Interfaces import IEnvironment
from Base import ObjectBase, MovableBase



class Brain(nn.Module):
    def __init__(self, n_landmark : int, n_agents : int, size_channel : int, hidden_size : int):
        super(Brain, self).__init__()
        self.lstm1 = nn.LSTM(2 * 2 * n_landmark + (n_agents - 1) * size_channel + 4, hidden_size, batch_first = True)
        self.seq = nn.Sequential(nn.Linear(hidden_size, 32), nn.ReLU(), nn.Linear(32, 3 + size_channel), nn.Sigmoid())
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.h = torch.zeros(1, 1, hidden_size).to(self.device)
        self.c = torch.zeros(1, 1, hidden_size).to(self.device)
        
    def forward(self, x):
        hs, (self.h, self.c) = self.lstm1(x, (self.h, self.c))
        out = self.seq(hs)
        return out


class Agent(MovableBase):
    def __init__(self, _id : int, agent_param : dict, n_agents : int, n_landmarks : int, size_chanel : int, size_episode : int, hidden_size: int, vel_const : float = 10):
        super().__init__(**agent_param)
        self.brain = Brain(n_landmarks, n_agents, size_chanel, hidden_size)
        self.size_chanel = size_chanel
        self.size_episode = size_episode
        self.episode = 0
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
        self.optimizer = torch.optim.Adam(self.brain.parameters(), lr = 0.005)
        self.optimizer.zero_grad()
        self.gamma = 0.995

    def stocastic_msg(self, msg : torch.Tensor):
        return toch.Tensor([1 if random() <= t else 0 for t in msg])

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
        self.states.append(event)
        self.brain.eval()
        with torch.no_grad():
            act = self.brain.forward(event.reshape(1, 1, -1)).reshape(-1)
            act = Bernoulli(act).sample()
            self.actions.append(act)
        self.__walk(act[0], act[1], act[2])  
        self.mensage = act[3:]
        self.__forward()
        self.__update_rewards()
        return act

    def store(self, file_path : str):
        torch.save(self.brain.state_dict(), file_path)
        
    def load(self, file_path : str):
        self.brain.load_state_dict(torch.load(file_path))
        self.brain.eval()

    def add_reward(self, r):
        if r > 0:
            self.has_learn = True
        self.reward += r
            
    def clear_memory(self):
        self.rewards = []
        self.actions = []
        self.states = []
        self.has_learn = False

    def __update_rewards(self):
        self.rewards.append(self.reward)
        self.reward = 0

    def learn(self, episode):
        if self.has_learn:
            self.brain.train()
            Gt = torch.tensor([np.sum(self.rewards[i:]*(self.gamma**(np.array(range(0, len(self.rewards) - i))))) for i in range(len(self.rewards))], requires_grad = False)            
            #plt.plot(range(len(Gt)), Gt)
            #plt.show()
            states = torch.stack(self.states, dim = 0).unsqueeze(0)
            action = self.brain(states)
            sampler = Bernoulli(action)
            log = sampler.log_prob(torch.stack(self.actions, dim = 0).unsqueeze(0)).sum(dim = 2)
            loss = torch.sum(Gt * log) 
            loss.backward()
            if episode % 10 == 0:
                self.optimizer.step()     
                self.optimizer.zero_grad()
            return loss.detach()
        else:
            return 0
        self.clear_memory()

class Landmark(MovableBase):
    def __init__(self, landmark_param : dict, thresold : float = 5):
        super().__init__(**landmark_param)
        self.thresold = thresold
        self.monitored_agents = []
        self.frozen = False
        self.time_frozen = 0

    def next_state(self):
        if self.thresold < self.acc.magnitude() and not self.frozen: 
            self.vel += self.acc * dt
        if self.frozen:
            self.time_frozen += 1
        super().next_state()
        
    def monitoring(self, agent : Agent):
        if agent not in self.monitored_agents:
            self.monitored_agents.append(agent)
        
    def reward_monitored(self, reward):
        if len(self.monitored_agents) > 1 and reward != 0:
            for agent in self.monitored_agents:
                agent.add_reward(reward)

    def clear_monitored(self):
        if self.vel.magnitude() <= 0.1:
            self.monitored_agents = []

    def draw(self, screen : Surface):
        dimensions = (self.pos.x - self.radius, self.pos.y - self.radius, self.diam, self.diam)
        ellipse(screen, CYAN if self.frozen else (self.color if self.vel.magnitude() <= 0.1 else YELLOW), dimensions, self.stroke_weight)

    def frozen_land(self):
        self.frozen = True

    def unfrozen(self):
        self.frozen = False
        self.time_frozen = 0


class Goal(ObjectBase):
    def __init__(self, goal_param : dict): 
        super().__init__(**goal_param)