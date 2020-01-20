# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 15:38:34 2020

@author: Emin
"""

import gym
import time

env = gym.make("Taxi-v3").env
env.render()

env.reset()

# %%

print("state space : ", env.observation_space)
print("Action space : ", env.action_space)

state = env.encode(4,3,3,3) 
print("State number : ", state)

env.s = state
env.render()

# %%

env.P[475]

# %%

env.reset()
time_step = 0
total_reward = 0
lise_visualize = []
while True:
    
    time_step += 1
    
    # choose action
    action = env.action_space.sample()
    
    # perform action and get reward
    state, reward, done, _ = env.step(action) # state = next state
    
    # total reward
    total_reward += reward
    
    # visualize
    lise_visualize.append({ "Frame" : env, "State" : state, "action" : action,
                            "reward": reward, "total reward" : total_reward })
    
    
    #env.render()
    
    
    
    if done:
        break
    
# %%
        
for i, frame in enumerate(lise_visualize):
    print(frame["Frame"].render())
    print("Timesetep : ", i+1)
    print("state : ", frame["State"])
    print("action : ", frame["action"])
    print("reward : ", frame["reward"])
    print("total reward : ", frame["total reward"])
    
    
    



    
    
    




















