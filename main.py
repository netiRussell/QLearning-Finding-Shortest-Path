# ! This file is a modification of the code provided in the following link: https://colab.research.google.com/drive/1E2RViy7xmor0mhqskZV14_NUj2jMpJz3#scrollTo=F1YO3mj_oS2J
# TODO: compare the models using fixed destination node and the same environment. Train both with the same time and compare the accuracy.

"""
  Goal of this file: train a reinforcement learning model to find the shortest path between two points given a
  50x50 grid environment and fixed destination node with the ID#60.
"""


import numpy as np
import math
import ast
import pandas as pd
import time

from supplemental import is_terminal_state
from supplemental import get_next_action 
from supplemental import get_next_location
from supplemental import get_shortest_path
from supplemental import convertToIDs

# TODO: delete
import sys

start_time = time.time()

"""
  Configurations
"""
environment_rows = 40
environment_columns = 40

# Defining q values holder
# a "Q-value" represents the expected future reward an agent can receive by taking a specific action ("a") in a particular state ("s") within an environment
# 4 dimensions because there is a maximum of 4 possible actions an agent can take from some state
q_values = np.zeros((environment_rows, environment_columns, 4))

# Actions
#numeric action codes: 0 = up, 1 = right, 2 = down, 3 = left
actions = ['up', 'right', 'down', 'left']

# Rewards
rewards = np.full((environment_rows, environment_columns), -1)
destination = 60

# Destination
rewards[math.floor(destination / environment_rows), destination % environment_rows] = 100

# Defining training parameters
epsilon = 0.9 #the percentage of time when we should take the best action (instead of a random action)
discount_factor = 0.9 #discount factor for future rewards
learning_rate = 0.9 #the rate at which the AI agent should learn






"""
  Training
"""

# Training loop
for epochID in range(5):
    # Read dataset
    df = pd.read_parquet("./40x40raw/40x40train.parquet", engine="auto")
    df = df.reset_index()

    print(f"Length: {len(df)}")
    
    print("Current epoch: ", epochID)
    
    for episodeID, row in df.iterrows():
      #get the starting location for this episode
    
      Y = list(ast.literal_eval(row['Y'].decode('utf-8'))[0])
      row_index = math.floor(Y[0] / environment_rows)
      column_index = Y[0] % environment_rows
    
      #continue taking actions (i.e., moving) until we reach a terminal state
      #(i.e., until we reach the item packaging area or crash into an item storage location)
      while not is_terminal_state(row_index, column_index, rewards):
        #choose which action to take (i.e., where to move next)
        action_index = get_next_action(row_index, column_index, epsilon, q_values)
    
        #perform the chosen action, and transition to the next state (i.e., move to the next location)
        old_row_index, old_column_index = row_index, column_index #store the old row and column indexes
        row_index, column_index = get_next_location(row_index, column_index, action_index, actions, size=environment_rows)
        
        #receive the reward for moving to the new state, and calculate the temporal difference
        reward = rewards[row_index, column_index]
        old_q_value = q_values[old_row_index, old_column_index, action_index]
        temporal_difference = reward + (discount_factor * np.max(q_values[row_index, column_index])) - old_q_value
    
        #update the Q-value for the previous state and action pair
        new_q_value = old_q_value + (learning_rate * temporal_difference)
        q_values[old_row_index, old_column_index, action_index] = new_q_value

print('Training complete!')


# Evaluate
results = []

# Read dataset
df = pd.read_parquet("./40x40raw/40x40eval.parquet", engine="auto")
df = df.reset_index()

# Evaluation loop
print(f"Eval length: {len(df)}")
for episodeID, row in df.iterrows():
  Y = list(ast.literal_eval(row['Y'].decode('utf-8'))[0])
  min_distance = len(Y)

  row_pair = math.floor(Y[0] / environment_rows)
  column_pair = Y[0] % environment_rows
  currentResult = convertToIDs(get_shortest_path(row_pair, column_pair, rewards, q_values, environment_columns, actions), environment_columns)
  
  if len(currentResult) == min_distance:
    if currentResult[-1] == destination:
      # print(f"Path#{row_pair*environment_columns + column_pair} is valid. \nPath: {currentResult}\nTarget: {Y}\n\n")
      results.append(1)


print(f"{(len(results) / (environment_columns**2)*0.8)*100}% is correct")
print(f"Wall-clock time taken to run the code: ", time.time() - start_time)
