import numpy as np
import random
from tqdm import tqdm
import os
import time
class laby(object):
    def __init__(self):
        super(laby, self).__init__()
        self.grid = [[-1,-1,-1,-1,-1,1,-1,-1],
                     [-1,0,0,0,0,0,0,-1],
                     [-1,-2,0,-1,-1,0,-1,-1],
                     [-1,0,0,-2,-1,0,-2,-1],
                     [-1,0,0,0,0,0,0,-1],
                     [-1,-1,-1,-1,0,-1,-1,-1],
                     [-1,0,0,0,0,0,0,-1],
                     [-1, -1, -1, -1, -1, -1, -1, -1]
                     ]
        self.y = 5
        self.x = 4
        self.action = [[0, 1],  # right
                       [-1, 0],  # Up
                       [0, -1],  # Left
                       [1, 0]  # down
                       ]

    def reset(self):
        self.y = 5
        self.x = 4
        return (self.y * 7 + self.x + 6)

    def is_finished(self):
        if self.grid[self.y][self.x] == 1:
            return True
        else:
            return False


    def step(self, actions):
        self.y = max(0, min(self.y + self.action[actions][0],7))
        self.x = max(0, min(self.x + self.action[actions][1],7))
        return (self.y*7 + self.x+1), self.grid[self.y][self.x]
        # return etat et reward


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
        self.mort = 0

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

    # Tue l'agent si il fait trop de merde
    def punishment(self, r):
        if r == -10:
            self.mort += 1
            return True
        else:
            return False
    def nbmort(self):
        return self.mort


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
            at = agent.takeaction(st,0.8)
            stp1, r = labyrinthe.step(at)
            punishment += r
            atp1 = agent.takeaction(stp1,0)
            agent.update(st,at,stp1,atp1)
            st = stp1
            #si l'agent fait trop de dinguerie il meurt ce batard
            if agent.punishment(punishment):
                break
    print("L'agent est mort: ", agent.nbmort())
    print("Total victoire: ", size - agent.nbmort())
    print("Pourcentage victoire:", ((size-agent.nbmort())/ size) *100,"%" )

    #Affichage de la Qtable
    Q_table = agent.Q
    #for i in range(1,64):
        #print("Etat", i ,":", Q_table[i])


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
    print("L'agent est mort: ", new_agent.nbmort())