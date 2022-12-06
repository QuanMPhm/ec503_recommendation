import numpy as np
import matplotlib.pyplot as plt


# What is the distribution of demographics, movie genres (for maybe analysis of bias in data later on)

# Files and initializations
n_genres = 19
n_users = 943
n_movies = 1682
n_data = 100000
f_dataset = open("IMDB_dataset/u.data", "r")
f_demo = open("IMDB_dataset/u.user", "r")
f_movies = open("IMDB_dataset/u.item", "r")
f_genre_names = open("IMDB_dataset/u.genre", "r")

#Table of data
data_table = list()
data_demo = list()
data_movies = list()
genre_names = list()

# Get all genre names
line = f_genre_names.readline()
while line:
    line = line.split("|")
    genre_names.append(line[0])
    line = f_genre_names.readline()

# Fill up data table
line = f_dataset.readline()
while line:
    line = line.split("\t")
    line = [int(i) for i in line]
    data_table.append(line)
    line = f_dataset.readline()

# Fill up demo table
line = f_demo.readline()
while line:
    line = line.split("|")
    line[0] = int(line[0])
    line[1] = int(line[1])
    line[4] = line[4].strip()
    data_demo.append(line)
    line = f_demo.readline()

# Fill up movies table
line = f_movies.readline()
is_int = [0] + list(range(5, 24))
while line:
    line = line.split("|")
    for i in is_int:
        line[i] = int(line[i])

    data_movies.append(line)
    line = f_movies.readline()

# Distribution of genres
genre_count = list()
for i in range(n_genres):
    genre_count.append(0)

for movie in data_movies:
    genre_score = movie[5:]
    genre_count = list(np.array(genre_count) + np.array(genre_score))

fig, ax = plt.subplots(1, 1)
ax.barh(genre_names, genre_count)
ax.set_title("Distribution of movie genres in dataset")
ax.set_xlabel('Movie count')
ax.set_ylabel('Genre')
plt.savefig("Data_Analysis_Plots/genre_distribution.jpg", dpi=200, bbox_inches="tight")

# Distribution of demo
# Age
age_count = list()
age_list = list()
# Get max age
max_age = 0
for i in data_demo:
    max_age = max(max_age, i[1])

for i in range(max_age):
    age_count.append(0)
    age_list.append(i)

for person in data_demo:
    age_count[person[1] - 1] += 1

fig, ax = plt.subplots(1, 1)
ax.barh(age_list, age_count)
ax.set_title("Distribution of age")
ax.set_xlabel('Person Count')
ax.set_ylabel('Age')
plt.savefig("Data_Analysis_Plots/age_distribution.jpg", dpi=200, bbox_inches="tight")

print(len(data_movies))

# Occupation, meh, later TODO

# Do users tend to only like specific genres, are from specific eras? Are there users who are generalists?
# For each user, we can get empirical mean the user will give certain rating for each genre.
# Lets say if the prob they gave a certain genre a 4-5 was +80% (Choice), we say they like that genre
# Lets also filter out genres that users have only watched less than 5 movies (heuristic chocie)
# Then for each user we can see how many genres they seem to like
users_genre_prob = []




# Do users with certain demographys like certain things? How can we ascertain this?