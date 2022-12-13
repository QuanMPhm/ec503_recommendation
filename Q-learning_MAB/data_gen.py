import numpy as np
import math

class User:
    def __init__(self, preferences, var) -> None:
        self.var = var
        self.preferences = preferences

    def recommend(self, movie):
        score = np.round(np.random.normal(self.preferences[movie], math.sqrt(self.var)))
        if score > 5: score = 5
        elif score < 1: score = 1
        return score
        

def gen_user(n_movies, var):
    '''
    Return the matrix of preference of size 5 x n_movies
    Each column contains the mean for a Gaussian distribution with variance var, mean ranging from 1 to 5
    We can then represent the user's rating for a given movie as belonging to a Gaussian distribution 
    '''
    preferences = [0] * n_movies
    preferences = list(np.random.uniform(0, 5, (n_movies,)))
    return User(preferences, var)


