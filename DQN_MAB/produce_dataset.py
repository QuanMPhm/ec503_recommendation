import numpy as np


# generate 10 samples with different preference
class generateSet:
    def __init__(self,numOfPerson=10000,numOfMAB=10):
        self.person = numOfPerson
        self.MAB = numOfMAB
        self.preference = np.zeros((numOfPerson, numOfMAB))
        self.randPreference()
        # NOT USE??
        self.sparse = 0.7   # may not click the content that did attract user

    ## Initialize random preference for all people
    def randPreference(self):
        for i in range(self.person):
            for j in range(self.MAB):
                self.preference[i,j] = np.random.rand()

    def renew_preference(self):
        pass

    def doULike(self,xthPerson=1,xthMAB=1):
        if np.random.rand() > self.preference[xthPerson,xthMAB]:
            return -1
        else:
            return 1

    def step(self,s,action):
        # The reward is the rank of the chosen action
        # I.e 0 is for the least preferred action
        # Max of 9 for the most preferred action
        reward=0
        for i in range(10):
            if i != action:
                if self.preference[s,i]<self.preference[s,action]:
                    reward+=1
        # todo: use better reward

        return reward