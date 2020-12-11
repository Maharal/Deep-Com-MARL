# Third part import
from random import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# Local application import
from PyGameFacade import *
from Interfaces import IEnvironment
from Base import MovableBase, ObjectBase
from Objects import Agent, Landmark, Goal


class Environment(IEnvironment):
    def __init__(self, 
            agent_param : dict, 
            landmark_param : dict, 
            goal_param : dict, 
            size_screen : tuple, 
            force_wall : float, 
            dist_wall : float, 
            eta : float,
            size_chanel : int, 
            size_epoch : int
        ):
        self.eta = eta
        self.size_screen = size_screen
        self.agents = [Agent(_id, agent_param["setup"], agent_param["num"], landmark_param["num"], size_chanel, size_epoch) for _id in range(agent_param["num"])]
        self.landmarks = self.__gen_without_intersec(Landmark, landmark_param["setup"], landmark_param["num"])
        self.goals = self.__gen_without_intersec(Goal, goal_param["setup"], goal_param["num"])
        self.force_wall = force_wall
        self.dist_wall = dist_wall
        self.num_steps = 4000
        self.step = 0
        self.epoch = 0
        self.gamma = 0.99
  
    def __has_has_intersection(self, items : list, item : ObjectBase) -> bool:
        for i in items:
            if i.has_intersection(item):
                return True
        return False

    def __gen_without_intersec(self, type_class : ObjectBase, param : dict, num : int) -> List[ObjectBase]:
        l = []
        for _ in range(num):
            new = type_class(param)
            while self.__has_has_intersection(l, new):
                new = type_class(param)
            l.append(new)
        return l

    def __apply_constrains(self, obj : MovableBase):
        width, height = self.size_screen
        obj.pos.x = constrain(obj.pos.x, 0, width-obj.diam)
        obj.pos.y = constrain(obj.pos.y, 0, height-obj.diam)

    def __wall_force(self, obj : MovableBase):
        width, height = self.size_screen
        force = Vector2(0, 0)
        if obj.pos.x < self.dist_wall:
            force.x += self.force_wall
        elif obj.pos.x > width - self.dist_wall:
            force.x -= self.force_wall
        if obj.pos.y < self.dist_wall:
            force.y += self.force_wall
        elif obj.pos.y > height - self.dist_wall:
            force.y -= self.force_wall
        obj.add_force(force)

    def __friction_force(self, obj : MovableBase):
        obj.vel += -self.eta * obj.vel

    def __collision_force(self, i : MovableBase, items : MovableBase, comp_reward = False):
        for j in items:
            if i != j and i.has_intersection(j):
                d_pos = i.pos - j.pos
                d_pos.scale_to_length(abs(i.radius + j.radius - d_pos.magnitude()))
                i.add_force(d_pos)
                if comp_reward:
                    j.monitoring(i)
                    if j.frozen:
                        i.add_reward(-1)

    def __fix_position(self, i : Landmark, items : MovableBase):
        for j in items:
            if i != j and i.has_intersection(j):
                d_pos = i.pos - j.pos
                d_pos.scale_to_length(abs(i.radius + j.radius - d_pos.magnitude()))
                i.pos += d_pos

    def __frozen(self, landmark):
        for goal in self.goals:
            if landmark.is_inside(goal) and not landmark.frozen:
                landmark.reward_monitored(10)
                landmark.frozen = True
                break    

    def __explore(self):
        for agent in self.agents:
            self.__apply_constrains(agent)
            self.__wall_force(agent)
            self.__collision_force(agent, self.landmarks, True)
            agent.R += -0.001
        for landmark in self.landmarks:
            self.__apply_constrains(landmark)
            self.__wall_force(landmark)
            self.__friction_force(landmark)
            self.__frozen(landmark)
            self.__collision_force(landmark, self.landmarks + self.agents)
        for movable in self.agents + self.landmarks:
            self.__fix_position(movable, self.agents + self.landmarks)
        mensages = []
        for agent in self.agents:
            msg = agent.next_state(self)
            mensages.append(msg)
        for landmark in self.landmarks:
            landmark.next_state()
            landmark.clear_monitored()

    def draw(self, screen : Surface):
        for obj in self.goals + self.agents + self.landmarks:
            obj.draw(screen)

    def next_state(self):
        self.step += 1
        if self.step < self.num_steps:
            self.__explore()
        else:
            self.step = 0
            self.epoch += 1
            print(f"EPOCH {self.epoch}: Start learning...")

            for agent in self.agents:
                print(agent.rewards)            

                ER = torch.tensor([np.sum(agent.rewards[i:]*(self.gamma**np.array(range(i, len(agent.rewards))))) for i in range(len(agent.rewards))])
                plt.plot(ER)
                plt.show()
            
            exit()
            print("end learning...")
            
    def load(self, path : str):
        for agent in self.agents:
            agent.load(f"{path}/model_{agent._id}.h5")
            
    def store(self, path : str):
        for agent in self.agents:
            agent.store(f"{path}/model_{agent._id}.h5")