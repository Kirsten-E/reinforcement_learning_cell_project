import numpy as np
from random import randint, random
import json
import matplotlib.pyplot as plt


def flatten(board):
    return str(tuple(cell for row in board for cell in row))

def square(x):
    return x * x

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

    
    def __init__(self, rows, columns):
        with open("basic_experimentation/dense_blobs/exploratory_value_function.json", 'r') as file:
            self.exploratory_value_function = json.load(file)
        with open("basic_experimentation/dense_blobs/feature_value_function.json", 'r') as file:
            self.feature_value_function = json.load(file)
        self.exploratory_param = 0.1
        self.exploratory_step_size = 0.1
        self.feature_param = 0.1
        self.feature_step_size = 0.1
        self.rows = rows
        self.columns = columns
        self.results = []
        self.successes = 0

    def explore_image(self, image : Image, iteration : int):
        self.x = 0
        self.y = 0
        self.explored_board = np.zeros((self.rows, self.columns))
        last_state_action = "blank"
        last_state_action_value = 0
        finished = False
        while not finished:
            if random() <= self.exploratory_param:
                action = randint(0, 4)
                finished = self.move(action)
                self.assign_prob(image)
            else:
                current_action = randint(0, 4)
                current_action_value = -4.5
                for action in range(5):
                    predicted_value = self.exploratory_value_function.get(f"{flatten(self.explored_board)}, {self.y}, {self.x}, {action}")
                    if not(predicted_value is None) and predicted_value > current_action_value:
                        current_action_value = predicted_value
                        current_action = action
                finished = self.move(current_action)
                if self.explored_board[self.y][self.x] == 0:
                    self.assign_prob(image)
                    self.exploratory_value_function[last_state_action] = last_state_action_value + self.exploratory_step_size * (1 + current_action_value - last_state_action_value)
                else:
                    self.exploratory_value_function[last_state_action] = last_state_action_value + self.exploratory_step_size * (current_action_value - last_state_action_value)
                last_state_action = f"{flatten(self.explored_board)}, {self.y}, {self.x}, {current_action}"
                last_state_action_value = current_action_value
        explored_elements = -1 * np.sum(square(image.board - self.explored_board))
        self.exploratory_value_function[last_state_action] = explored_elements
        file = open("basic_experimentation/dense_blobs/results.txt", 'a')
        file.write(f"{iteration}: {explored_elements}\n")
        file.close()
        self.results.append(explored_elements)
        print(iteration)
        print(self.explored_board)
        print(explored_elements)
        if explored_elements == 0:
            self.successes += 1


    def assign_prob(self, image):
        if random() <= self.feature_param:
            self.explored_board[self.y][self.x] = round(random(), 2)
        else:
            predicted_value = self.feature_value_function.get(f"{image.board[self.y][self.x]}")
            if predicted_value is None:
                predicted_value = 0.5
            self.explored_board[self.y][self.x] = predicted_value

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
        with open("basic_experimentation/dense_blobs/exploratory_value_function.json", "w") as file:
            json.dump({str(k): v for k, v in self.exploratory_value_function.items()}, file, indent=2)
        with open("basic_experimentation/dense_blobs/feature_value_function.json", "w") as file:
            json.dump({str(k): v for k, v in self.feature_value_function.items()}, file, indent=2)



image = Image(5, 5, 20)
agent = Agent(5, 5)

for i in range(0, 100):
    agent.explore_image(image, i)
agent.save_value_function()

print(image.board)
plt.plot(range(0,100), agent.results)
plt.show()
print(agent.successes)

