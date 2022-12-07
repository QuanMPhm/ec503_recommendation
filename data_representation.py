import numpy as np
import datetime
import time

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
f_orig_embeddings = open("listwise recommend/embeddings.csv")

# Read in embeddings
data_orig_embeddings = list()
line = f_orig_embeddings.readline() # Skip first line
line = f_orig_embeddings.readline()
while line:
    line = line.split(";")
    movie_embedding = line[1].split("|")
    movie_embedding = [float(i) for i in movie_embedding]
    data_orig_embeddings.append([int(line[0]), movie_embedding])
    line = f_orig_embeddings.readline()

# Fill up movies table
data_movies = list()
line = f_movies.readline()
is_int = [0] + list(range(5, 24))
while line:
    line = line.split("|")
    for i in is_int:
        line[i] = int(line[i])

    data_movies.append(line)
    line = f_movies.readline()

# Get min time to normalize movie release times
month_map = {
    "Jan" : 1,
    "Feb" : 2, 
    "Mar" : 3, 
    "Apr" : 4, 
    "May" : 5, 
    "Jun" : 6, 
    "Jul" : 7, 
    "Aug" : 8, 
    "Sep" : 9, 
    "Oct" : 10, 
    "Nov" : 11, 
    "Dec" : 12
}

min_time = -1
for movie in data_movies:

    # In case no date given
    if not movie[2]:
        continue

    movie_time = movie[2].split('-')
    movie_t_day = int(movie_time[0])
    movie_t_month = movie_time[1]
    movie_t_year = int(movie_time[2]) + 50

    # Obtain UNIX time in hours
    movie_t_month = month_map[movie_t_month]
    date_time = datetime.datetime(movie_t_year, movie_t_month, movie_t_day, 0, 0)
    movie_t = time.mktime(date_time.timetuple()) / 3600

    if min_time == -1:
        min_time = movie_t
    elif min_time > movie_t:
        min_time = movie_t

# Get user rating history...


def state_pad_no_udata(s):
    """Pad the state"""
    return 0

def step_no_udata(s, a, r, n):
    """gets the next step that does not include user data"""
    movie = data_movies[a - 1]
    if not movie[2]:
        movie_t = 0
    else:
        movie_time = movie[2].split('-')
        movie_t_day = int(movie_time[0])
        movie_t_month = movie_time[1]
        movie_t_year = int(movie_time[2]) + 50

        # Obtain UNIX time in hours
        movie_t_month = month_map[movie_t_month]
        date_time = datetime.datetime(movie_t_year, movie_t_month, movie_t_day, 0, 0)
        movie_t = time.mktime(date_time.timetuple()) / 3600 - min_time


    # Next state is some function of previous state, action, and reward
    # Next state is a vector containing s = [shifted UNIX time, 19 1 and 0s for genre indicator] * n
    # So a representation is a n * 20 vector of the n most recent movie reconmendations
    s_next = list()
    for i in range(n):
        s_next.append([0] * (n_genres + 1 + 1)) # +1 for time, +1 for rating

    # Shift previous state back
    s_next[1:] = s[:(len(s) - 1)]
    s_next[0] = [movie_t] + movie[5:] + [r]
    return s_next

def step_udata(s, a, r, n):
    """gets the next step that does include user data"""
    # Next state is some function of previous state, action, and our user boi
    # Next state is a vector containing s = [shifted UNIX time, 19 1 and 0s for genre indicator] * n
    # Also add one vector at top to represent user gender, year of birth or age, occupation as encoded
    # So a representation is a n * 20 vector of the n most recent movie reconmendations
    movie = data_movies[a - 1]
    if not movie[2]:
        movie_t = 0
    else:
        movie_time = movie[2].split('-')
        movie_t_day = int(movie_time[0])
        movie_t_month = movie_time[1]
        movie_t_year = int(movie_time[2]) + 50

        # Obtain UNIX time in hours
        movie_t_month = month_map[movie_t_month]
        date_time = datetime.datetime(movie_t_year, movie_t_month, movie_t_day, 0, 0)
        movie_t = time.mktime(date_time.timetuple()) / 3600 - min_time


    s_next = list()
    for i in range(n + 1): # +1 for user date
        s_next.append([0] * (n_genres + 1 + 1)) # +1 for time, +1 for rating

    # Shift previous state back
    s_next[2:] = s[1:(len(s) - 1)]
    s_next[1] = [movie_t] + movie[5:] + [r]
    s_next[0] = s[0] # Assume user data supplied at start

    return s_next

def step_other(s, a, r, uid):
    """gets the next step based on what the original code did"""
    # Which was given the user, just randomly grab the (shitty) embeddings for 10 random movies they rated in the past
    # Will need to read up on embeddings

    # In this shitty case, we will need uid
    # Let's say in this case, its so shitty we won't do this for now. I'll say original author is an idiot
    # TODO
    return 0

def step_word2vec_embed(s, a, r, n):
    
    return 0

# s = list()
# for i in range(10 + 1):
#     s.append([0] * (1 + n_genres + 1))

# s[0] = [25, 1]

# a = step_udata(s, 1, 5, 10)
# print(step_udata(a, 300, 4, 10))
# print(step_udata)