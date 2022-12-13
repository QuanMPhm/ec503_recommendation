import numpy as np
import datetime
import time
import pickle

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
f_genre_names = open("IMDB_dataset/u.copy.genre", "r") # The modified genre names
# f_orig_embeddings = open("listwise recommend/embeddings.csv")

genre_names = list()
embed_genre = dict()
embed_titles = dict()
# Read in word2vec embeddings
with open("genre_embed.pickle", 'rb') as f:
    embed_genre = pickle.load(f)

with open("IMDB_embed.pickle", 'rb') as f:
    embed_titles = pickle.load(f)

# # Read in shitty embeddings
# embed_orig = list()
# line = f_orig_embeddings.readline() # Skip first line
# line = f_orig_embeddings.readline()
# while line:
#     line = line.split(";")
#     movie_embedding = line[1].split("|")
#     movie_embedding = [float(i) for i in movie_embedding]
#     embed_orig.append([int(line[0]), movie_embedding])
#     line = f_orig_embeddings.readline()
# f_orig_embeddings.close()

# Get all genre names
line = f_genre_names.readline()
while line:
    line = line.split("|")
    genre_names.append(line[0])
    line = f_genre_names.readline()

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
f_movies.close()

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

    # Extract scaled UNIX time of release
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

def step_w2v(s, a, r, n):
    # a is movie id
    # State is text embedding of n most recent rated movies and their ratings
    # s_next is n x (embed_size + 1 + 1)
    embed_size = 300
    
    movie = data_movies[a - 1]
    # Extract scaled UNIX time of release
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

    # Get embedding, first extract genres, and words in movie title
    movie_title = movie[1].split()
    movie_title_filtered = list()
    for word in movie_title:
        word = word.lower()
        if "(" in word:
            continue
        word = ''.join(c for c in word if c.isalnum())
        movie_title_filtered.append(word)
    
    movie_genres = list()
    genres_liked = np.where(np.array(movie[5:]) == 1)[0]
    for igenre in genres_liked:
        movie_genres.append(genre_names[igenre].lower())

    # Then weighted average embeddings
    m_embedding = np.array([0.] * embed_size)
    # IMPORTANT. These weights tell how much more our embeddings care about genre than titles
    title_w = 1.
    genre_w = 3.
    sum_w = 0

    for word in movie_title_filtered:
        if word not in embed_titles.keys():
            continue
        
        w_embed = list()
        for i in embed_titles[word]:
            w_embed.append(i[0])
        m_embedding += title_w * np.array(w_embed)
        sum_w += title_w
    
    for g in movie_genres:
        w_embed = list()
        for i in embed_genre[g]:
            w_embed.append(i[0])
        m_embedding += genre_w * np.array(w_embed)
        sum_w += genre_w
    
    m_embedding = list(m_embedding / sum_w)

    s_next = list()
    for i in range(n):
        s_next.append([0] * (embed_size + 1 + 1)) # +1 for rating, +1 for year of release
    # Shift previous state back
    s_next[1:] = s[:(len(s) - 1)]
    s_next[0] = [movie_t] + m_embedding + [r]
    return s_next


embed_size = 300
raw_size = n_genres + 2
s = list()
# for i in range(10):
#     s.append([0] * (1 + embed_size + 1))

for i in range(10):
    s.append([0] * raw_size)


print(step_no_udata(s, 300, 4, 10))