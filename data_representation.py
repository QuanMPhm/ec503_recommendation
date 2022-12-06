import numpy as np

# Given state, action, and reward recieved, transition to new state
# action would be the id of the movie that was picked
# reward assume to be some number that we then map to a rating 1-5

# Files and initializations
n_genres = 19 # Number of genres
n_users = 943
n_movies = 1682
n_data = 100000
f_dataset = open("IMDB_dataset/u.data", "r")
f_demo = open("IMDB_dataset/u.user", "r")
f_movies = open("IMDB_dataset/u.item", "r")
f_genre_names = open("IMDB_dataset/u.genre", "r")

data_movies = list()

# Fill up movies table
line = f_movies.readline()
is_int = [0] + list(range(5, 24))
while line:
    line = line.split("|")
    for i in is_int:
        line[i] = int(line[i])

    data_movies.append(line)
    line = f_movies.readline()



def step_no_udata(s, a, r):
    """gets the next step that does not include user data"""
    movie = data_movies[a - 1]

    # Next state is some function of previous state, action, and reward
    s_next = ""
    return s_next

def step_udata(s, a, r):
    """gets the next step that does include user data"""
    return 0

def step_other(s, a, r):
    """gets the next step based on what the original code did"""
    return 0