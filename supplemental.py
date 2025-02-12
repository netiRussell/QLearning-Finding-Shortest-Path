"""
  Here are the supplemental functions needed for the main program
"""

import numpy as np

# Defining a function that determines if the specified location is a terminal state
def is_terminal_state(current_row_index, current_column_index, rewards):
  #if the reward for this location is -1, then it is not a terminal state (i.e., it is a 'white square')
  if rewards[current_row_index, current_column_index] == -1.:
    return False
  else:
    return True

# Defining an epsilon greedy algorithm that will choose which action to take next (i.e., where to move next)
def get_next_action(current_row_index, current_column_index, epsilon, q_values):
  #if a randomly chosen value between 0 and 1 is less than epsilon, 
  #then choose the most promising value from the Q-table for this state.
  if np.random.random() < epsilon:
    # Choose the action with the highest q value for this state
    return np.argmax(q_values[current_row_index, current_column_index])
  else: #choose a random action
    return np.random.randint(4)

# Defining a function that will get the next location based on the chosen action
def get_next_location(current_row_index, current_column_index, action_index, actions, size):
  new_row_index = current_row_index
  new_column_index = current_column_index
  if actions[action_index] == 'up' and current_row_index > 0:
    new_row_index -= 1
  elif actions[action_index] == 'right' and current_column_index < size - 1:
    new_column_index += 1
  elif actions[action_index] == 'down' and current_row_index < size - 1:
    new_row_index += 1
  elif actions[action_index] == 'left' and current_column_index > 0:
    new_column_index -= 1
  return new_row_index, new_column_index

# Defining a function that will get the shortest path between any location within the warehouse that 
# Agent is allowed to travel and the item packaging location.
def get_shortest_path(start_row_index, start_column_index, rewards, q_values, size, actions):
  #return immediately if this is an invalid starting location
  if is_terminal_state(start_row_index, start_column_index, rewards):
    return []
  else: #if this is a 'legal' starting location
    current_row_index, current_column_index = start_row_index, start_column_index
    shortest_path = []
    shortest_path.append([current_row_index, current_column_index])
    #continue moving along the path until we reach the goal (i.e., the item packaging location)
    while not is_terminal_state(current_row_index, current_column_index, rewards) and len(shortest_path) < size:
      #get the best action to take
      action_index = get_next_action(current_row_index, current_column_index, 1, q_values)
      #move to the next location on the path, and add the new location to the list
      current_row_index, current_column_index = get_next_location(current_row_index, current_column_index, action_index, actions, size)
      shortest_path.append([current_row_index, current_column_index])
    return shortest_path

# Defining a function that convert pair-like coordinates into IDs of the nodes 
def convertToIDs(shortest_path, size):
  final_path = []
  for pair in shortest_path:
    final_path.append(pair[0]*size + pair[1])
  
  return final_path