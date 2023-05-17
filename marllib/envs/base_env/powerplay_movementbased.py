
import gym
from gym.spaces import Discrete, MultiDiscrete, Dict, Box
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.tune.utils import merge_dicts
import networkx as nx
import typing
import math
from itertools import islice
import numpy as np
import random as rand

policy_mapping_dict = {
    "all_scenario": {
        "description": "powerplay",
        "team_prefix": ("red_", "blue_"),
        "all_agents_one_policy": False,
        "one_agent_one_policy": False,
    }
}

class Powerplay(MultiAgentEnv):

    def __init__(self, env_config):
        self.junctions = {
            0: ConeStation("terminal", [0]),
            1: ConeStation("terminal", [5]),
            2: ConeStation("terminal", [30]),
            3: ConeStation("terminal", [35]),
            4: ConeStation("junction", [0,1,6,7], junction_type = "GROUND"),
            5: ConeStation("junction", [1,2,7,8], junction_type = "LOW"),
            6: ConeStation("junction", [2,3,8,9], junction_type = "GROUND"),
            7: ConeStation("junction", [3,4,9,10], junction_type = "LOW"),
            8: ConeStation("junction", [4,5,10,11], junction_type = "GROUND"),
            9: ConeStation("junction", [6,7,12,13], junction_type= "LOW"),
            10: ConeStation("junction", [7,8,13,14], junction_type= "MEDIUM"),
            11: ConeStation("junction", [8,9,14,15], junction_type= "HIGH"),
            12: ConeStation("junction", [9,10,15,16], junction_type= "MEDIUM"),
            13: ConeStation("junction", [10,11,16,17], junction_type= "LOW"),
            14: ConeStation("junction", [12,13,18,19], junction_type = "GROUND"),
            15: ConeStation("junction", [13,14,19,20], junction_type = "HIGH"),
            16: ConeStation("junction", [14,15,20,21], junction_type = "GROUND"),
            17: ConeStation("junction", [15,16,21,22], junction_type = "HIGH"),
            18: ConeStation("junction", [16,17,22,23], junction_type = "GROUND"),
            19: ConeStation("junction", [18,19,24,25], junction_type= "LOW"),
            20: ConeStation("junction", [19,20,25,26], junction_type= "MEDIUM"),
            21: ConeStation("junction", [20,21,26,27], junction_type= "HIGH"),
            22: ConeStation("junction", [21,22,27,28], junction_type= "MEDIUM"),
            23: ConeStation("junction", [22,23,28,29], junction_type= "LOW"),
            24: ConeStation("junction", [24,25,30,31], junction_type = "GROUND"),
            25: ConeStation("junction", [25,26,31,32], junction_type = "LOW"),
            26: ConeStation("junction", [26,27,32,33], junction_type = "GROUND"),
            27: ConeStation("junction", [27,28,33,34], junction_type = "LOW"),
            28: ConeStation("junction", [28,29,34,35], junction_type = "GROUND"),
        }#every junction and their corresponding ID. IDs will be used to clean up code by a lot
        
        self.blue_points = 0
        self.red_points = 0
        self.blue_substation = ConeStation("substation", [12, 18])

        self.red_substation = ConeStation("substation", [17, 23])

        self.blue_stack_one = ConeStation("stack", [2])

        self.blue_stack_two = ConeStation("stack", [32])

        self.red_stack_one = ConeStation("stack", [3])

        self.red_stack_two = ConeStation("stack", [33])

        self.red_beacon_one_placed = False
        self.blue_beacon_one_placed = False
        self.red_beacon_two_placed = False
        self.blue_beacon_two_placed = False
        self.red_terminal_one_captured = False
        self.blue_terminal_one_captured = False
        self.red_terminal_two_captured = False
        self.blue_terminal_two_captured = False
        

        
        
        seed = 1
        rand.seed(seed)
        self.agents = ["red_1", "red_2", "blue_1", "blue_2"]
        self.robots = {"red_1": Robot("red_1", "red", 11, 180, rand.randint(1, 3)),
                        "red_2": Robot("red_2", "red", 23, 180, rand.randint(1,3)),
                        "blue_1": Robot("blue_1", "blue", 6, 0, rand.randint(1,3)),
                        "blue_2": Robot("blue_2", "blue", 24, 0, rand.randint(1,3))}#creates the robots mapped to the agent names
        self.action_type = {"red_1": "choice",
                            "red_2": "choice",
                            "blue_1": "choice",
                            "blue_2": "choice"}#Types of actions: junction_choice, cone_choice, to_cones, waiting
                                                
        self.action_space = Discrete(11)
        #can be action masked down: choices are [left, right, forward, backward, junction_topleft, junction_topright, 
        #junction_botleft, junction_botright, terminal, pick_up_cone, pick_up_beacon]
        temp = [241, 2, 30, 36, 4, 2, 2, 20, 36, 4, 2, 2, 20, 36, 4, 2, 2, 20, 36, 4, 2, 2, 20, 40, 40, 2, 2,2, 2, 30, 30, 2, 2, 2, 2]
        
        b = Box(low=np.array([0]*(29+len(temp))), high=np.array([2]*29 +temp), dtype=np.float32)
        self.observation_space = Dict({'obs': b,
                                    'action_mask': Box(low = 0, high = 1, shape = (11,),  dtype=np.float32)})
        self.env_config = env_config
        self.time_elapsed = 0
        


    def get_env_info(self):
        env_info = {
            "space_obs": self.observation_space,
            "space_act": self.action_space,
            "num_agents": 4,
            "episode_limit": 400,
            "policy_mapping_info": policy_mapping_dict
        }
        return env_info
    def reset(self):
        self.__init__({"seed": rand.randint(1,10000)})
        observations = {i: self.generate_observation(i) for i in self.agents}
        return observations
    def step(self, action_dict):
        #print("STEPPPED")
        rewards = {"red_1": 0, "red_2": 0, "blue_1": 0, "blue_2": 0}
        obs = {}
        for a in self.agents:
            # if(a != "red_1"):
            #     continue
            robot_ptr = self.robots[a]
            if(a in action_dict): action = action_dict[a]
            else: action = None
            if(self.action_type[a] == "choice"):#if the bot's last decision was a junction choice
                #we find it optimized to ignore the case in which the coach tells the driver to stop placing a cone on a junction
                #follow this path if there is stuff to follow
                if(action == 0):
                    robot_ptr.box_location -= 1
                    rewards[a] -= 1
                elif(action == 1):
                    robot_ptr.box_location +=1
                    rewards[a] -= 1
                elif(action == 2):
                    robot_ptr.box_location -= 6
                    rewards[a] -= 1
                elif(action == 3):
                    robot_ptr.box_location +=6
                    rewards[a] -= 1
                elif(action == 4 or action == 5 or action == 6 or action == 7):
                    self.action_type[a] = "automated"
                    robot_ptr.adjusting = True
                    robot_ptr.adjustingTime = robot_ptr.adjustmentPeriod
                    topleft = -1
                    topright = -1
                    bottomleft = -1
                    bottomright = -1
                    for j in self.junctions:
                        if(j<=3): continue
                        if(self.junctions[j].box_locations[0] == robot_ptr.box_location):
                            bottomright = j
                        elif(self.junctions[j].box_locations[1] == robot_ptr.box_location):
                            bottomleft = j
                        elif(self.junctions[j].box_locations[2] == robot_ptr.box_location):
                            topright = j
                        elif(self.junctions[j].box_locations[3] == robot_ptr.box_location):
                            topleft = j
                    if(action == 4):
                        robot_ptr.junction_to = topleft
                    elif(action == 5):
                        robot_ptr.junction_to = topright
                    elif(action == 6):
                        robot_ptr.junction_to = bottomleft
                    elif(action == 7):
                        robot_ptr.junction_to = bottomright
                    
                    
                        
                elif(action == 8):
                    if(robot_ptr.box_location == 0):
                        if(not self.red_terminal_one_captured):
                            rewards[a] += 3
                        else:
                            rewards[a] -= 1
                        self.red_terminal_one_captured = True
                    elif(robot_ptr.box_location == 5):
                        if(not self.blue_terminal_one_captured):
                            rewards[a] += 3
                        else:
                            rewards[a] -= 1
                        self.blue_terminal_one_captured = True
                    elif(robot_ptr.box_location == 30):
                        if(not self.blue_terminal_two_captured):
                            rewards[a] += 3
                        else:
                            rewards[a] -= 1
                        self.blue_terminal_two_captured = True
                    elif(robot_ptr.box_location == 35):
                        if(not self.red_terminal_two_captured):
                            rewards[a] += 3
                        else:
                            rewards[a] -= 1
                        self.red_terminal_two_captured = True
                    if(robot_ptr.team == "red"):
                        self.red_points += 1
                    else:
                        self.blue_points +=1

                elif(action == 9):
                    substationAt = self.findSubstation(a)
                    robot_ptr.holdingCone = True
                    substationAt.cones_held -= 1

                elif(action == 10):
                    robot_ptr.holdingBeacon = True
                    substationAt = self.findSubstation(a)
                    substationAt.cones_held -= 1
                
            
            elif(self.action_type[a] == "automated"):
                
                if(robot_ptr.crashed and robot_ptr.adjustingTime  != 0 and not(robot_ptr.adjusting)):#if it got into a crash(detected by the computer at the end of each step)
                    robot_ptr.adjustingTime -= 0.5
                    if(robot_ptr.adjustingTime == 0):#if its free to keep going
                        self.action_type[a] = "choice"
                elif(robot_ptr.adjusting):#if it's putting a cone on a pole
                    robot_ptr.adjustingTime -= 0.5
                    
                    if(robot_ptr.adjustingTime == 0):
                        
                        junc = self.junctions[robot_ptr.junction_to]
                        if(junc.beaconed):
                            rewards[a] -= 5
                        else:
                            if(robot_ptr.holdingBeacon):
                                rewards[a] += 6
                                junc.beaconed = True
                                robot_ptr.gotBeacon = True
                            rewards[a] += junc.determine_value()
                            junc.owned_by = robot_ptr.team
                            if(robot_ptr.team == "red"):
                                self.red_points += junc.determine_value()
                            else:
                                self.blue_points += junc.determine_value()
                        self.action_type[a] = "choice"
                        robot_ptr.junction_to = None
                        robot_ptr.adjusting = False
                        robot_ptr.crashed = False

            
        #determine which robots have crashed; only works on 2 at a time, though it is important to recognize the unlikeliness of 3 colliding at once
        for a in self.agents:
            for b in self.agents:
                if(a == b):
                    continue
                elif(self.robots[a].crashed or self.robots[b].crashed):
                    continue
                elif(self.robots[a].box_location == self.robots[b].box_location and not(self.robots[a].adjusting) and not(self.robots[b].adjusting)):
                    self.robots[a].crashed = True
                    self.robots[b].crashed = True
                    self.robots[a].adjustingTime = 1
                    self.robots[b].adjustingTime = 1
                    self.action_type[a] = "automated"
                    self.action_type[b] = "automated"


        #determine the rewards
        #delete rewards for agents that aren't doing anything atm, since their actions are governed by computers
        for a in self.agents:
            if(not(self.time_elapsed < 120 and self.action_type[a] == "automated" and rewards[a] == 0)):
                obs[a] = self.generate_observation(a)
            
        dones = {a : False for a in self.agents}
        dones["__all__"] = False

        


        #final rewards if the game is over
        if(self.time_elapsed >= 120):
            dones = {a : True for a in self.agents}
            dones["__all__"] = True
            red_p, blue_p, red_circ, blue_circ = self.calcFinalPoints()
            red_p = self.red_points + 20*int(red_circ)
            blue_p = self.blue_points + 20*int(blue_circ)

            rewards["red_1"] += red_p / 5
            rewards["red_2"] += red_p / 5
            rewards["blue_1"] += blue_p / 5
            rewards["blue_2"] += blue_p / 5

            if (self.robots["red_1"].adjustmentPeriod + self.robots["red_2"].adjustmentPeriod
                    < self.robots["blue_1"].adjustmentPeriod + self.robots["blue_2"].adjustmentPeriod):  # deals with cycle times and circuiting
                if (red_p < blue_p):  # lower cycle time = expected to win. if they don't, they get heavily penalized
                    rewards["red_1"] -= blue_p / 5
                    rewards["red_2"] -= blue_p / 5

                else:
                    rewards["red_1"] -= blue_p / 10  # if they get a higher score with higher cycle time, not as big of penalties
                    rewards["red_2"] -= blue_p / 10
                    rewards["blue_1"] -= red_p / 10  # blue still has to get better tho, even if they got unlucky w/ random cycle times
                    rewards["blue_2"] -= red_p / 10
                if (not red_circ):  # lower cycle time = expectation to circuit
                    rewards["red_1"] -= 13
                    rewards["red_2"] -= 13
                    rewards["blue_1"] += 10
                    rewards["blue_2"] += 10  # reward for defense
                else:
                    rewards["red_1"] += 15
                    rewards["red_2"] += 15
                    rewards["blue_1"] -= 8
                    rewards["blue_2"] -= 8  # penalty for not defending, higher than 5 to ensure that the bot defends
                if (blue_circ):  # lower cycle time = expectation to defend
                    rewards["red_1"] -= 7
                    rewards["red_2"] -= 7
                    rewards["blue_1"] += 15
                    rewards["blue_2"] += 15
                else:
                    rewards["red_1"] += 8  # reward for defense is still high, even w/ low cycle to ensure bot defends
                    rewards["red_2"] += 8
                    rewards["blue_1"] -= 7
                    rewards["blue_2"] -= 7
            else:
                if (blue_p < red_p):
                    rewards["blue_1"] -= red_p / 5
                    rewards["blue_2"] -= red_p / 5
                else:
                    rewards["blue_1"] -= red_p / 10
                    rewards["blue_2"] -= red_p / 10
                rewards["red_1"] -= blue_p / 10
                rewards["red_2"] -= blue_p / 10
                if (not blue_circ):  # lower cycle time = expectation to circuit
                    rewards["blue_1"] -= 15
                    rewards["blue_2"] -= 15
                    rewards["red_1"] += 10
                    rewards["red_2"] += 10
                else:
                    rewards["blue_1"] += 5
                    rewards["blue_2"] += 5
                    rewards["red_1"] -= 5
                    rewards["red_2"] -= 5
                if (red_circ):  # lower cycle time = expectation to defend
                    rewards["blue_1"] -= 7
                    rewards["blue_2"] -= 7
                    rewards["red_1"] += 15  # high reward b/c they somehow circuited w/ higher cycle time
                    rewards["red_2"] += 15
                else:
                    rewards["blue_1"] += 7  # reward for defense
                    rewards["blue_2"] += 7
                    rewards["red_1"] -= 5
                    rewards["red_2"] -= 5
            self.agents = []
        self.time_elapsed += 0.5
        #print("obs type: " + str(np.dtype(obs)) + ", rewards dtype: " + str(np.dtype(rewards)) + " dones dtype: " + str(np.dtype(dones)))
        return obs, rewards, dones, {}
        
                
    


    def findSubstation(self, botName):
        robot_ptr = self.robots[botName]
        if(robot_ptr.team == "red"):
            if(robot_ptr.box_location in self.red_stack_one.box_locations):
                return self.red_stack_one
            if(robot_ptr.box_location in self.red_stack_two.box_locations):
                return self.red_stack_two
            if(robot_ptr.box_location in self.red_substation.box_locations):
                return self.red_substation
        elif(robot_ptr.team == "blue"):
            if(robot_ptr.box_location in self.blue_stack_one.box_locations):
                return self.blue_stack_one
            if(robot_ptr.box_location in self.blue_stack_two.box_locations):
                return self.blue_stack_two
            if(robot_ptr.box_location in self.blue_substation.box_locations):
                return self.blue_substation
        return None
    def calcFinalPoints(self):
        red_points = self.red_points
        blue_points = self.blue_points
        red_circuit, blue_circuit = self.find_circuit()
        for i in range(4, 29):  # ownership calcs
            if (self.junctions[i].owned_by == "red" and not(self.junctions[i].beaconed)): #+10 was counted early to emphasize beacons; also the +10 is guaranteed so like :shrug:
                red_points += 3
            elif (self.junctions[i].owned_by == "blue" and not(self.junctions[i].beaconed)):
                blue_points += 3
        return red_points + 20 * int(red_circuit), blue_points + 20 * int(blue_circuit), red_circuit, blue_circuit
    def find_circuit(self):#simple bfses
        red_cir = False
        if (self.red_terminal_one_captured and self.red_terminal_two_captured):
            stack = [4, 5, 9]
            searched = []
            while (len(stack) > 0):
                val = stack.pop(0)
                if (val in searched):
                    continue
                else:
                    searched.append(val)
                if (val < 4 or val >= 29):
                    continue
                if (self.junctions[val].owned_by == "red"):
                    if (val in [23, 27, 28]):
                        red_cir = True
                    else:
                        val_abs = val - 4  # makes it easier to work with since val will be 4-29
                        if (not (val_abs % 5 == 0)):  # means its not on left edge
                            stack.append(val - 1)  # one left
                        if (not (val_abs % 5 == 4)):  # not on right edge
                            stack.append(val + 1)  # one right
                        if (not (val_abs + 5 >= 25)):  # not on bottom edge
                            stack.append(val + 5)  # one down
                        if (not (val_abs - 5 < 0)):  # not on top edge
                            stack.append(val - 5)  # one up
                        if (not (val_abs % 5 == 0) and not (val_abs - 5 < 0)):
                            stack.append(val - 6)  # diag upper left
                        if (not (val_abs % 5 == 4) and not (val_abs - 5 < 0)):
                            stack.append(val - 4)  # diag upper right
                        if (not (val_abs % 5 == 0) and not (val_abs + 5 >= 25)):
                            stack.append(val + 4)  # diag lower left
                        if (not (val_abs % 5 == 4) and not (val_abs + 5 >= 25)):
                            stack.append(val + 6)  # diag lower right
        blue_circ = False
        
        if(self.blue_terminal_one_captured and self.blue_terminal_two_captured):
            stack = [7, 8, 13]
            searched = []
            while (len(stack) > 0):
                val = stack.pop(0)
                if (val in searched):
                    continue
                else:
                    searched.append(val)

                if (val < 4 or val >= 29):
                    continue
                if (self.junctions[val] == -1):
                    if (val in [19, 24, 25]):
                        blue_circ = True
                        break
                    else:
                        val_abs = val - 4  # makes it easier to work with since val will be 4-29
                        if (not (val_abs % 5 == 0)):  # means its not on left edge
                            stack.append(val - 1)  # one left
                        if (not (val_abs % 5 == 4)):  # not on right edge
                            stack.append(val + 1)  # one right
                        if (not (val_abs + 5 >= 25)):  # not on bottom edge
                            stack.append(val + 5)  # one down
                        if (not (val_abs - 5 < 0)):  # not on top edge
                            stack.append(val - 5)  # one up
                        if (not (val_abs % 5 == 0) and not (val_abs - 5 < 0)):
                            stack.append(val - 6)  # diag upper left
                        if (not (val_abs % 5 == 4) and not (val_abs - 5 < 0)):
                            stack.append(val - 4)  # diag upper right
                        if (not (val_abs % 5 == 0) and not (val_abs + 5 >= 25)):
                            stack.append(val + 4)  # diag lower left
                        if (not (val_abs % 5 == 4) and not (val_abs + 5 >= 25)):
                            stack.append(val + 6)  # diag lower right
        return red_cir, blue_circ
    def close(self):
        x = 0

    def generate_observation(self, botName):
        action_mask = [1]*11
        robot_ptr = self.robots[botName]

        #determine valid moves
        #choices are [left, right, forward, backward, junction_topleft, junction_topright, junction_botleft, junction_botright, terminal, pick_up_cone, pick_up_beacon]
        if(robot_ptr.box_location % 6 == 0):
            action_mask[0] = 0
            action_mask[4] = 0
            action_mask[6] = 0
        if(robot_ptr.box_location % 6 == 5):
            action_mask[1] = 0
            action_mask[5] = 0
            action_mask[7] = 0
        if(robot_ptr.box_location < 6):#first row
            action_mask[2] = 0
            action_mask[4] = 0
            action_mask[5] = 0
        if(robot_ptr.box_location >= 30):#last row
            action_mask[3] = 0
            action_mask[6] = 0
            action_mask[7] = 0
        if(not(robot_ptr.holdingBeacon or robot_ptr.holdingCone)):
            action_mask[4] = 0
            action_mask[5] = 0
            action_mask[6] = 0
            action_mask[7] = 0
            action_mask[8] = 0
            substationAt = self.findSubstation(botName)
            if(substationAt == None or substationAt.cones_held == 0):
                action_mask[9] = 0
                action_mask[10] = 0
        if(robot_ptr.holdingBeacon or robot_ptr.holdingCone):
            action_mask[9] = 0
            action_mask[10] = 0
        if(robot_ptr.adjusting):
            action_mask = [0]*11
        if(self.time_elapsed <= 90 or robot_ptr.gotBeacon):
            action_mask[10] = 0
        
        junction_owns = []
        for a in self.junctions:
            if(self.junctions[a].owned_by is None):
                junction_owns.append(0)
            else:
                junction_owns.append(int(self.junctions[a].owned_by == "red")+1)
        current_obs = junction_owns + [int(self.time_elapsed),
                                                         int(self.robots[botName].team == "red"),
                                                         int(abs(self.robots[botName].junction_to + 1) if self.robots[botName].junction_to is not None else 0),
                                                         int(self.robots["red_1"].box_location),
                                                         int(self.robots["red_1"].heading / 90.0),
                                                         int(self.robots["red_1"].holdingCone),
                                                         int(self.robots["red_1"].holdingBeacon),
                                                         int(self.robots["red_1"].adjustmentPeriod * 2),
                                                         int(self.robots["red_2"].box_location),
                                                         int(self.robots["red_2"].heading / 90.0),
                                                         int(self.robots["red_2"].holdingCone),
                                                         int(self.robots["red_2"].holdingBeacon),
                                                         int(self.robots["red_2"].adjustmentPeriod * 2),
                                                         int(self.robots["blue_1"].box_location),
                                                         int(self.robots["blue_1"].heading / 90.0),
                                                         int(self.robots["blue_1"].holdingCone),
                                                         int(self.robots["blue_1"].holdingBeacon),
                                                         int(self.robots["blue_1"].adjustmentPeriod * 2),
                                                         int(self.robots["blue_2"].box_location),
                                                         int(self.robots["blue_2"].heading / 90.0),
                                                         int(self.robots["blue_2"].holdingCone),
                                                         int(self.robots["blue_2"].holdingBeacon),
                                                         int(self.robots["blue_2"].adjustmentPeriod * 2),
                                                         int(self.blue_points / 10),
                                                         int(self.red_points / 10),
                                                         int(self.red_beacon_one_placed+1),
                                                         int(self.red_beacon_two_placed+1),
                                                         int(self.blue_beacon_one_placed+1),
                                                         int(self.blue_beacon_two_placed+1),
                                                         int(self.blue_substation.cones_held + self.blue_stack_one.cones_held + self.blue_stack_two.cones_held),
                                                         int(self.red_substation.cones_held + self.red_stack_one.cones_held + self.red_stack_two.cones_held),
                                                         int(self.red_terminal_one_captured),
                                                         int(self.blue_terminal_one_captured),
                                                         int(self.red_terminal_two_captured),
                                                         int(self.blue_terminal_two_captured)
                                                         ]
        return {"obs": np.array(current_obs, dtype=np.float32), "action_mask": np.array(action_mask, dtype=np.float32)}

    def render(self):

        print("CUR TIME:" + str(self.time_elapsed))
        #if (self.render_mode == "debug"):
        print("STATE FOR GRAPH BELOW: " + str(self.red_points) + " ," + str(self.blue_points))
        mask = self.generate_observation("red_1")["action_mask"]
        print("ACTION MASK OF RED_1" + str(mask))
        spacesNeededLeft = 2 if self.blue_substation.cones_held >= 10 else 1
        spacesNeededRight = 2 if self.red_substation.cones_held >= 10 else 1
        initial = (str(int(self.junctions[0].owned_by == "red")) + spacesNeededLeft * " " + "   " + str(
            self.blue_stack_one.cones_held) + " " + str(
            self.red_stack_one.cones_held) + spacesNeededRight * " " + "   " + str(int(self.junctions[1].owned_by == "red")) + "\n" +
                    spacesNeededLeft * " " + "_ _ _ _ _ _" + spacesNeededRight * " " + "\n" +
                    spacesNeededLeft * " " + " G L G L G " + spacesNeededRight * " " + "\n" +
                    spacesNeededLeft * " " + "_ _ _ _ _ _" + spacesNeededRight * " " + "\n" +
                    spacesNeededLeft * " " + " L M H M L " + spacesNeededRight * " " + "\n" +
                    spacesNeededLeft * " " + "_ _ _ _ _ _" + spacesNeededRight * " " + "\n" +
                    str(self.blue_substation.cones_held) + " G H G H G " + str(self.red_substation.cones_held) + "\n" +
                    spacesNeededLeft * " " + "_ _ _ _ _ _" + spacesNeededRight * " " + "\n" +
                    spacesNeededLeft * " " + " L M H M L " + spacesNeededRight * " " + "\n" +
                    spacesNeededLeft * " " + "_ _ _ _ _ _" + spacesNeededRight * " " + "\n" +
                    spacesNeededLeft * " " + " G L G L G " + spacesNeededRight * " " + "\n" +
                    spacesNeededLeft * " " + "_ _ _ _ _ _" + spacesNeededRight * " " + "\n" +
                    str(int(self.junctions[2].owned_by == "red")) + spacesNeededLeft * " " + "   " + str(
                    self.blue_stack_two.cones_held) + " " + str(
                    self.red_stack_two.cones_held) + spacesNeededRight * " " + "   " + str(int(self.junctions[3].owned_by == "red")))

        for bot in self.robots:
            box_y = math.floor(self.robots[bot].box_location/6)
            box_x = self.robots[bot].box_location - box_y*6
            x = box_x * 2 + spacesNeededLeft
            y = box_y * 2 + 1
            ind = y * (12 + spacesNeededRight + spacesNeededLeft) + x
            toPut = "R" if self.robots[bot].team == "red" else "B"
            initial = initial[:ind] + toPut + initial[ind + 1:]
        return initial

class Robot():
    def __init__(self, name, team, box_location, heading, adjustmentPeriod):
        self.name = name
        self.team = team
        self.box_location = box_location
        self.heading = heading
        self.adjustmentPeriod = adjustmentPeriod
        self.gotBeacon = False
        self.holdingBeacon = False
        self.holdingCone = False
        self.adjusting = False
        self.adjustingTime = 0
        self.path = None
        self.junction_to = None
        self.crashed = False

    def __str__(self):
        return "crashed: " + str(self.crashed) + ", adjusting: " + str(self.adjusting)
        
        


        

class ConeStation(): #either junction or cone station; realistically, they both store cones
    def __init__(self, type : str, box_locations : list, junction_type = "NONE"):
        self.box_locations = box_locations
        self.owned_by = None
        self.beaconed = False
        self.junction_type = junction_type#only really used when determining junctions, not when determining cone stations
        self.type = type
        if(type == "junction"):
            self.cones_held = 0
        elif(type == "substation"):
            self.cones_held = 20
        elif(type == "stack"):
            self.cones_held = 5
        elif(type == "terminal"):
            self.cones_held = 0
    
    def determine_value(self):
        if(self.junction_type == "HIGH"):
            return 5
        elif(self.junction_type == "MEDIUM"):
            return 4
        elif(self.junction_type == "LOW"):
            return 3
        elif(self.junction_type == "GROUND"):
            return 2
    
if __name__ == "__main__":
    pp = Powerplay({"seed":1})
    for i in range(100):
        print(pp.render())
        action = int(input("STEP #" + str(i) + ": "))
        pp.step({"red_1": action})



    


        

    
