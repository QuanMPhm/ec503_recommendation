import numpy as np
import math
# generate 10 samples with different preference
class generateSet:
    def __init__(self,numOfPerson=10000,numOfMAB=10,number_he_like=10,method='random',reward_method = 'more than'):
        self.person = numOfPerson
        self.number_he_like = number_he_like
        self.MAB = numOfMAB
        self.preference = np.zeros((numOfPerson, numOfMAB))
        self.randPreference(method)
        self.sparse = 0.8   # may not click the content that did attract user
        self.reward_method = reward_method

    # Different methods of initializations
    # Neat 
    def randPreference(self,method = 'random'):
        if method == 'random':
            for i in range(self.person):
                for j in range(self.MAB):
                    self.preference[i, j] = np.random.rand()
        elif method == 'like&hate':
            for i in range(self.person):
                # TODO: Wouldn't this result in duplicates?
                index_like = np.random.randint(1, self.MAB,[1,self.number_he_like])
                for j in range(self.number_he_like):
                    self.preference[i,index_like[0][j]] = np.random.randint(math.floor(0.9*self.MAB*10),self.MAB*10)/(self.MAB*10)
        elif method == 'like&ok':
            for i in range(self.person):
                index_like = np.random.randint(1, self.MAB, [1, self.number_he_like])
                for j in range(self.number_he_like):
                    self.preference[i, index_like[0][j]] = np.random.randint(math.floor(0.9 * self.MAB * 10),
                                                                             self.MAB * 10) / (self.MAB * 10)
                for j in range(self.MAB):
                    if self.preference[i, j] == 0:
                        self.preference[i, j] = np.random.randint(1, math.floor(0.9 * self.MAB * 10)) / (self.MAB * 10)
        else:
            for i in range(self.person):
                for j in range(self.MAB):
                    self.preference[i, j] = np.random.rand()

    def renew_preference(self):
        pass

    def doULike(self,xthPerson=1,xthMAB=1):
        if np.random.rand() > self.preference[xthPerson,xthMAB]:
            return -1
        else:
            return 1

    def step(self,person_choose,s,action,actions_value,reward_method='more than'):
        reward=0
        reward_method = self.reward_method
        # cosine similarity:
        actions_value = np.array(actions_value).flatten()
        # print('actions_value111111',actions_value)
        # print('np.min(a)',np.min(actions_value))
        if np.min(actions_value)<0:
            actions_value = actions_value - np.min(actions_value)
        # print('actions_value22222',actions_value)
        # print('sum11111',np.sum(actions_value))
        actions_value = actions_value/np.sum(actions_value)

        s = np.array(s).flatten()
        if reward_method == 'cosine_linear':
            cosine_similarity = np.dot(actions_value,s)/(np.linalg.norm(actions_value, 2) * np.linalg.norm(s, 2))
            # todo: regularize
            reward = cosine_similarity
            # print('reward:',reward)
        elif reward_method == 'cosine_exp':
            cosine_similarity = np.dot(actions_value,s)/(np.linalg.norm(actions_value, 2) * np.linalg.norm(s, 2))
            # todo: regularize
            reward = (np.exp(cosine_similarity)-1.6)*3
            #reward = cosine_similarity
            # print('reward:',reward)
        elif reward_method == 'more than':
            for i in range(self.MAB):
                if i != action:
                    if self.preference[person_choose,i]<self.preference[person_choose,action]:
                        reward+=1

            cosine_similarity = 0
        else:
            print('reward_method error')
            cosine_similarity = 0
        return reward,cosine_similarity,actions_value