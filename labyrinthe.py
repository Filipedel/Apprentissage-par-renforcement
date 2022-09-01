import numpy as np
import random
from tqdm import tqdm
import os
import time
class laby(object):
    def __init__(self):
        super(laby, self).__init__()
        #Maze, -1 is a wall, 1 is the exit , 0 is the path and -2 is a trap
        self.grid = [[-1,-1,1,-1,-1,-1,-1,-1],
                     [-1,0,0,0,0,0,0,-1],
                     [-1,-2,0,-1,-1,0,-1,-1],
                     [-1,0,0,-2,-1,0,-2,-1],
                     [-1,0,0,0,0,0,0,-1],
                     [-1,-1,-1,-1,0,-1,-1,-1],
                     [-1,0,0,0,0,0,0,-1],
                     [-1, -1, -1, -1, -1, -1, -1, -1]
                     ]
        #Position from the scratch
        self.y = 6
        self.x = 1
        # the robot can use only  this list of actions
        self.action = [[0, 1],  # right
                       [-1, 0],  # Up
                       [0, -1],  # Left
                       [1, 0]  # down
                       ]

    def reset(self):
        self.y = 6
        self.x = 1
        return (self.y * 7 + self.x + 1)

    def is_finished(self):
        if self.grid[self.y][self.x] == 1:
            return True
        else:
            return False


    def step(self, actions):
        self.y = max(0, min(self.y + self.action[actions][0],7))
        self.x = max(0, min(self.x + self.action[actions][1],7))
        return (self.y*7 + self.x+1), self.grid[self.y][self.x]
        # return state and reward


    def show(self):
        print("-------------------")
        y = 0
        for line in self.grid:
            x = 0
            for pt in line:
                print("%s\t" % (pt if y != self.y or x != self.x else "X"), end="")
                x += 1
            y += 1
            print("")




class Robot(object):
    def __init__(self):
        super(Robot, self).__init__()
        #Qtable
        self.Q = [[0,0,0,0], [0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],
                  [0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],
                  [0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],
                  [0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],
                  [0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],
                  [0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],
                  [0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],
                  [0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],
                  [0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],
                  [0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],
                  [0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
        self.death = 0

    def takeaction(self,state,eps):
        #epsilon greedy
        if random.uniform(0,1) < eps:
            action = np.random.randint(0,4)
        else:
            action = np.argmax(self.Q[state])
        return action

    #test robot fin
    def testrealaction(self,state,Q):
        action = np.argmax(Q[state])
        return action

    def update(self,state, action, stateprime, actionprime, learning_rate=0.1, gamma=0.9):
        self.Q[state][action] = self.Q[state][action] + learning_rate * (r + gamma*(self.Q[stateprime][actionprime]) - self.Q[state][action])

    # Kill the robot if he looses too many rewards and count the number of death
    def punishment(self, r):
        if r == -10:
            self.death += 1
            return True
        else:
            return False
    def nbmort(self):
        return self.death


if __name__=="__main__":
    labyrinthe = laby()
    agent = Robot()
    st = labyrinthe.reset()
    size = 10000
    #Training
    for i in tqdm(range(size + 1)):
        st = labyrinthe.reset()
        punishment = 0
        while  labyrinthe.is_finished() is not True:
            at = agent.takeaction(st,0.9)
            stp1, r = labyrinthe.step(at)
            punishment += r
            atp1 = agent.takeaction(stp1,0)
            agent.update(st,at,stp1,atp1)
            st = stp1
            if agent.punishment(punishment):
                break
    print("The agent is dead: ", agent.nbmort()," times")
    print("Win percentage in training:", ((size-agent.nbmort())/ size) *100,"%" )

    #Show the Q table
    Q_table = agent.Q

    #Test
    new_agent = Robot()
    new_state = labyrinthe.reset()

    while labyrinthe.is_finished() is not True:
        labyrinthe.show()
        at = new_agent.testrealaction(new_state,Q_table)
        stp2, r2 = labyrinthe.step(at)
        new_state = stp2
        time.sleep(1)
        os.system('cls')
    (print(" Agent is dead") if new_agent.nbmort() > 0 else print(" Agent Alive and victorious"))
    for i in range(1,64):
        print("State", i ,":", Q_table[i])