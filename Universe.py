# Standard import
from random import random, shuffle
from typing import List
from math import pi, sin, cos

# Third part import
from numpy.core.fromnumeric import size
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# Local application import
from PyGameFacade import constrain, Vector2, Surface
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
            size_channel : int, 
            num_steps : int,
            time_penality : float,
            hit : float,
            frozen_hit :  float,
            push : float
        ):
        self.eta = eta
        self.size_screen = size_screen
        self.force_wall = force_wall
        self.dist_wall = dist_wall
        self.num_steps = num_steps
        self.step = 0
        self.episode = 0
        self.agents = [Agent(_id, agent_param["setup"], agent_param["num"], landmark_param["num"], size_channel, num_steps, agent_param["hiden_state"]) for _id in range(agent_param["num"])]
        self.landmarks = self.__gen_without_intersec(Landmark, landmark_param["setup"], landmark_param["num"])
        self.goals = self.__gen_without_intersec(Goal, goal_param["setup"], goal_param["num"])
        self.time_penality = time_penality
        self.hit = hit
        self.frozen_hit = frozen_hit
        self.ground = goal_param["setup"]["diam"]/4
        self.ceiling = size_screen[0]/2
        self.temperature = self.ground
        self.push = push
        self.__support(self.landmarks, self.goals, self.temperature)
        self.loss = []
        
    def __has_has_intersection(self, items : list, item : ObjectBase) -> bool:
        for i in items:
            if i.has_intersection(item) and i is not item:
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

    def __support(self, landmarks : ObjectBase, goals : ObjectBase, r : float):
        index = list(range(len(goals))) 
        shuffle(index)
        for i in range(len(goals)):
            root = goals[index[i]].pos
            teta = 2*pi*random()
            landmarks[i].pos = Vector2(root.x + r * cos(teta), root.y + r * sin(teta))
            landmarks[i].unfrozen()

    def __reset_without_intersec(self) -> List[ObjectBase]:
        for goal in self.goals:
            goal.new_pos()
            while self.__has_has_intersection(self.goals, goal):
                goal.new_pos()
        for agent in self.agents:
            agent.new_pos()
            while self.__has_has_intersection(self.agents, agent):
                agent.new_pos()
        self.__support(self.landmarks, self.goals, self.temperature)

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
                if d_pos.magnitude() > 0:
                    d_pos.scale_to_length(abs(i.radius + j.radius - d_pos.magnitude()))
                    i.add_force(d_pos)
                if comp_reward:
                    j.reward_monitored(self.push)
                    j.monitoring(i)
                    if j.frozen and j.time_frozen < 100:
                        j.reward_monitored(self.frozen_hit)

    def __fix_position(self, i : Landmark, items : MovableBase):
        for j in items:
            if i != j and i.has_intersection(j):
                d_pos = i.pos - j.pos
                if d_pos.magnitude() > 0:
                    d_pos.scale_to_length(abs(i.radius + j.radius - d_pos.magnitude()))
                i.pos += d_pos

    def __frozen(self, landmark):
        for goal in self.goals:
            if landmark.is_inside(goal) and not landmark.frozen:
                landmark.reward_monitored(self.hit)
                landmark.frozen_land()
                break    

    def __explore(self):
        for agent in self.agents:
            self.__apply_constrains(agent)
            self.__wall_force(agent)
            self.__collision_force(agent, self.landmarks, True)
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
            agent.add_reward(self.time_penality)
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
            self.episode += 1
            l = np.mean([agent.learn(self.episode) for agent in self.agents])
            print(f"episode {self.episode}: Loss: {l}")
            self.loss.append(l)
            self.__reset_without_intersec()
            if sum([l.frozen for l in self.landmarks]) >= 3:
                self.temperature = min(1.05 * self.temperature, self.ceiling)
            elif sum([l.frozen for l in self.landmarks]) == 0:
                self.temperature = max(0.9 * self.temperature, self.ground)

        for l in self.landmarks:
            is_inside_goal = False
            for goal in self.goals:
                if l.is_inside(goal):
                    is_inside_goal = True
                    break
            if not is_inside_goal and l.frozen:
                l.unfrozen()

    def load(self, path : str):
        self.step = 0
        for agent in self.agents:
            agent.load(f"{path}/model_{agent._id}.h5")
            agent.clear_memory()
            
    def store(self, path : str):
        self.step = 0
        for agent in self.agents:
            agent.clear_memory()
            agent.store(f"{path}/model_{agent._id}.h5")