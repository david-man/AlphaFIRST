
from gym.spaces import Discrete, MultiDiscrete, Dict, Box
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.tune.utils import merge_dicts
import networkx as nx
import typing
import math
from itertools import islice
import numpy as np
import random as rand

from mecanum import Mecanum
import shapely
from shapely import Point, Polygon
from marllib import marl
from marllib.envs.base_env import ENV_REGISTRY
import time


policy_mapping_dict = {
    "all_scenario": {
        "description": "mecanum_movement",
        "all_agents_one_policy": True,
        "one_agent_one_policy": True,
    }
}
class MecMvmt(MultiAgentEnv):
    def __init__(self, env_config):
       
        self.env_config = env_config
        self.env_min_x = env_config["min_x"]
        self.env_max_x = env_config["max_x"]
        self.env_min_y = env_config["min_y"]
        self.env_max_y = env_config["max_y"]
        sample_space = Box(low = np.array([self.env_min_x, self.env_min_y, 0]), 
                           high = np.array([self.env_max_x, self.env_max_y, 2*math.pi]), dtype = np.float32)
        self.action_space = Box(low = 0.0, high = 1.0, shape = (1,), dtype = np.float32)
        self.observation_space = Dict({"obs": Box(low = np.array([-10000.0, -10000.0, -10000.0, -10000.0, -10000.0, 0]), 
                                     high = np.array([10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 1000]), dtype = np.float32)})# NEEDS CHANGE
        self.robot = Mecanum(max_motor_velocity = float(env_config["max_motor_velocity"]),
                             motor_acceleration= float(env_config["motor_acceleration"]),
                             wheel_radius=float(env_config["wheel_radius"]),
                             track_width=float(env_config["track_width"]),
                             wheelbase_length=float(env_config["wheelbase_length"]),
                             start_point = sample_space.sample())
        self.agents = ["power", "strafe", "turn"]
        self.num_agents = len(self.agents)
        
        self.goal_point = Point(sample_space.sample()[:-1])
        self.roadblocks = []
        self.curtime = float(0.0)
        self.maxtime = env_config["max_time"]# 0 < maxtime < 1000
        self.timestep = env_config["timestep"]
        

    def reset(self):
        self.__init__(self.env_config)
        obs = {}
        for agent in self.agents:
            obs[agent] = self.generate_obs()
        return obs

    def generate_obs(self):
        return {"obs": np.array([self.robot.x,
                         self.robot.y,
                         self.robot.theta,
                         self.goal_point.coords.xy[0][0],
                         self.goal_point.coords.xy[1][0],
                         self.curtime], 
                         dtype = np.float32)}
    
    def get_env_info(self):
        env_info = {
            "space_obs": self.observation_space,
            "space_act": self.action_space,
            "num_agents": self.num_agents,
            "episode_limit": 10000,
            "policy_mapping_info": policy_mapping_dict
        }
        return env_info

    def step(self, action_dict):
        '''THIS ONLY HANDLES THE MOVEMENT ACTIONS AND IS RESERVED FOR MEC DRIVES, NOT ANY OTHER GAME-SPECIFIC OR ROBOT-SPECIFIC ACTION. '''
        
        rewards = {}
        #print(action_dict)
        dones = {"__all__": False}
        self.robot.move(power = float(action_dict["power"][0]), strafe = float(action_dict["strafe"][0]), turn = float(action_dict["turn"][0]), timestep=self.timestep)
        if(self.curtime == self.maxtime):
            rewards ={"power":-100.0, "strafe":-100.0, "turn":-100.0}
            dones = {"__all__": True}
        elif True in (self.robot.clips_polygons(self.roadblocks)):
            self.robot.revert()
            rewards ={"power":-10.0, "strafe":-10.0, "turn":-10.0}
        elif self.robot.out_of_bounds(bounds = [Point(self.env_min_x, self.env_min_y),
                                      Point(self.env_min_x, self.env_max_y),
                                      Point(self.env_max_x, self.env_min_y),
                                      Point(self.env_max_x, self.env_max_y)]):
            self.robot.revert()
            rewards ={"power":-1.0, "strafe":-1.0, "turn":-1.0}
        elif self.robot.point_contained(self.goal_point):
            rewards ={"power":50.0, "strafe":50.0, "turn":50.0}
            dones = {"__all__": True}
        else:
            rewards ={"power":-0.5, "strafe":-0.5, "turn":-0.5}
        obs = {"power": self.generate_obs(), "strafe": self.generate_obs(), "turn": self.generate_obs()}
        
        self.curtime += 1
        
        #print(self.render())
        return obs, rewards, dones, {}
    
    def render(self):
        return "Current location: " + str(self.robot.x) + ", " + str(self.robot.y) + "\n. The goal is at " + str(self.goal_point.coords.xy[0][0]) + ", " + str(self.goal_point.coords.xy[1][0])
        
if __name__ == '__main__':
    global timestep
    timestep = 0
    # register new env
    ENV_REGISTRY["mec_gym"] = MecMvmt
    # initialize env
    env = marl.make_env(environment_name="mec_gym", map_name="trial1")
    # pick mappo algorithms
    mappo = marl.algos.mappo(hyperparam_source="common")
    # customize model
    model = marl.build_model(env, mappo, {"core_arch": "gru", "encode_layer": "128-128"})
    # start learning
    mappo.fit(env, model, stop={'timesteps_total': 1}, local_mode=False, num_gpus=1, share_policy='all', checkpoint_freq=5, checkpoint_end=True)
        

        

