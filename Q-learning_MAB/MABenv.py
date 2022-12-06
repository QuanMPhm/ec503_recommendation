import numpy as np
from numpy import random

class MAB:
    def __init__(self,numOfMAB,learning_rate=0.01, reward_decay=0.9):
        self.Q = np.zeros(numOfMAB)
        #self.win = np.arange(numOfMAB)/numOfMAB
        #self.win = np.array([0.1,0.2,0.3,0.2,0.1,0.2,0.4,0.3,0.4,0.8])

        # Hard coded winning probabilities :((
        self.win = np.array([0.1,0.2,0.3,0.2,0.1,0.2,0.4,0.3,0.7,0.8])

        # Counter of number of pulls for each machine
        self.pull = np.zeros(numOfMAB)
        # Unused??
        self.alpha = 0. 
        self.lr = learning_rate
        self.rd = reward_decay


    def step(self,action):
        result = random.rand()
        self.pull[action] = self.pull[action]+1
        if result<self.win[action]:
            r = 0.1
        else:
            r = -0.1
        self.learn(a=action,s=1,r=r,s_=1)


    def learn(self,a,s=1,r=0.0,s_=1):
        # This equation...
        self.Q[a] += self.lr * (r + self.rd*max(self.Q) - self.Q[a])
