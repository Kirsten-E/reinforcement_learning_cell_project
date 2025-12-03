import numpy as np
from random import randint, random
import json
import matplotlib.pyplot as plt
import multiprocessing
from math import sqrt


def flatten(board):
    return str(tuple(cell for row in board for cell in row))


class Image():

    def __init__(self, rows, columns, n):
        '''
        Generate an x by y 'image' with n singular blobs
        --- Parameters ---
            rows : Int
                Number of rows
            columns : Int
                Number of columns
            n : Int
                Number of blobs
        '''
        assert (rows*columns >= n)
        self.board = np.zeros((rows,columns))
        for _ in range(n):
            placed = False
            while not placed:
                position = randint(0, rows*columns-1)
                i = position // columns
                j = position % columns
                if self.board[i][j] == 0:
                    self.board[i][j] = 1
                    placed = True
        print(self.board)

class Agent():

    
    def __init__(self, rows, columns, id):
        self.id = id
        with open(f"multi_agent_experimentation/sync_at_end/exploratory_value_function_{self.id}.json", 'r+') as file:
            self.exploratory_value_function = json.load(file)
        with open(f"multi_agent_experimentation/sync_at_end/feature_value_function_{self.id}.json", 'r+') as file:
            self.feature_value_function = json.load(file)
        self.exploratory_param = 0.1
        self.exploratory_step_size = 0.1
        self.feature_param = 0.1
        self.feature_step_size = 0.1
        self.rows = rows
        self.columns = columns

    def explore_image(self, image : Image, iteration : int):
        # Flow of execution: Agent moves -> updates the value at that point if unvisited -> updates the value function -> marks as explored
        # Initialize a training iteration
        #(board_at_start,action_made) = (board_at_start,action_made) + a[(board_next,action_made_next) + reward_for_action_made_next - (board_at_start,action_made)]
        self.x = randint(0, self.rows-1)
        self.y = randint(0, self.columns-1)
        self.board_of_exploration = np.zeros((self.rows, self.columns))
        self.board_of_guesses = np.zeros((self.rows, self.columns))
        # Initialise dummy values for the first iteration
        last_state_action = f"{flatten(self.board_of_exploration)}, {self.x}, {self.y}, start"
        last_state_action_value = 0
        finished = False
        while not finished:
            # Move agent
            if random() <= self.exploratory_param:
                # Move agent
                current_action = randint(0, 4)
                current_state_action = f"{flatten(self.board_of_exploration)}, {self.x}, {self.y}, {current_action}"
                current_action_value = self.exploratory_value_function.get(current_state_action)
                if current_action_value is None:
                    current_action_value = -1 * (np.sum(self.board_of_exploration) / 2)
                finished = self.move(current_action)
            else:
                # Select action randomly with a guess of half of the explored elements are incorrect
                current_action = randint(0, 4)
                current_action_value = self.exploratory_value_function.get(f"{flatten(self.board_of_exploration)}, {self.x}, {self.y}, {current_action}")
                if current_action_value is None:
                    current_action_value = -1 * (np.sum(self.board_of_exploration) / 2)
                for action in range(5):
                    predicted_value = self.exploratory_value_function.get(f"{flatten(self.board_of_exploration)}, {self.x}, {self.y}, {action}")
                    # Minimise the penalty
                    if not(predicted_value is None) and predicted_value > current_action_value:
                        current_action_value = predicted_value
                        current_action = action
                # Move agent
                current_state_action = f"{flatten(self.board_of_exploration)}, {self.y}, {self.x}, {current_action}"
                finished = self.move(current_action)
            if self.board_of_exploration[self.y][self.x] == 0:
                # Update the value at that point
                self.assign_prob(image)
                # Update value function
                self.exploratory_value_function[last_state_action] = last_state_action_value + self.exploratory_step_size * (-1 * abs(image.board[self.y][self.x] - self.board_of_guesses[self.y][self.x]) + current_action_value - last_state_action_value)
            else:
                # Update value function
                self.exploratory_value_function[last_state_action] = last_state_action_value + self.exploratory_step_size * (current_action_value - last_state_action_value)
            last_state_action = current_state_action
            last_state_action_value = current_action_value
            # Mark as explored
            self.board_of_exploration[self.y][self.x] = 1
        # Calculate final result
        explored_elements = -1 * np.sum(abs(image.board - self.board_of_guesses))
        # Update final state
        self.exploratory_value_function[last_state_action] = explored_elements
        # Write results
        file = open(f"multi_agent_experimentation/sync_at_end/results_{self.id}.txt", 'a+')
        file.write(f"{iteration}: {explored_elements}\n")
        file.close()
        # print(iteration)
        # print(self.board_of_exploration)
        # print(self.board_of_guesses)
        # print(explored_elements)


    def assign_prob(self, image):
        if random() <= self.feature_param:
            self.board_of_guesses[self.y][self.x] = random()
        else:
            predicted_value = self.feature_value_function.get(f"{image.board[self.y][self.x]}")
            if predicted_value is None:
                predicted_value = 0.5 + 0.1 * random()
            self.board_of_guesses[self.y][self.x] = predicted_value
            self.feature_value_function[image.board[self.y][self.x]] = predicted_value + self.exploratory_step_size * (image.board[self.y][self.x] - predicted_value)


    def move(self, action):
        finished = False
        match action:
            case 0: # North
                if self.y == 0:
                    self.y = self.rows - 1
                else:
                    self.y = self.y - 1
            case 1: # East
                self.x = (self.x + 1) % self.columns
            case 2: # South
                self.y = (self.y + 1) % self.rows
            case 3: # West
                if self.x == 0:
                    self.x = self.columns - 1
                else:
                    self.x = self.x - 1
            case 4:# Finish
                finished = True
        return finished



    def save_value_function(self):
        with open(f"multi_agent_experimentation/sync_at_end/exploratory_value_function_{self.id}.json", "w+") as file:
            json.dump({str(k): v for k, v in self.exploratory_value_function.items()}, file, indent=2)
        with open(f"multi_agent_experimentation/sync_at_end/feature_value_function_{self.id}.json", "w+") as file:
            json.dump({str(k): v for k, v in self.feature_value_function.items()}, file, indent=2)

def train(agent, image):
    for i in range(0, 100000):
        agent.explore_image(image, i)
        # print(agent.id)
    agent.save_value_function()



if __name__ == '__main__':
    image_1 = Image(5, 5, 20)
    agent_1 = Agent(5, 5, "1")
    agent_2 = Agent(5, 5, "2")

    p1 = multiprocessing.Process(target=train, args=(agent_1, image_1))
    p2 = multiprocessing.Process(target=train, args=(agent_2, image_1))

    p1.start()
    p2.start()

    p1.join()
    p2.join()

    print("Done")

    print(image_1.board)
    agent_1.explore_image(image_1, 101)
    print(agent_1.board_of_guesses)
    agent_2.explore_image(image_1, 101)
    print(agent_2.board_of_guesses)


    print(np.minimum(agent_1.board_of_guesses, agent_2.board_of_guesses))
