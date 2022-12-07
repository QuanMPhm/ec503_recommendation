'''This file reads from the Google word2vec dataset, containing pretrained word embeddings for their stupidly large corpus
It then extracts the word embeddings for every word in every movie of the IMDB dataset, as well as the embeddings for the 
movie genre names

Extraction of the embeddings from the raw binary files was possible by reverse engineering the Github code for word2vec
Some genres were renamed to more generic names: Sci-Fi -> Fiction, Film-Noir -> Noir, etc since they were not found in dataset

Genre names adjustments found in u.copy.genre

Google word2vec dataset: https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing
Github wrd2vec: https://github.com/tmikolov/word2vec
'''
import struct
import pickle

f = open("C:/Users/Quan Minh Pham/Documents/BU/Classes/2022 Fall/ec503/GoogleNews-vectors-negative300.bin", "rb")
words = int(f.read(8)) # Number of words
embed_len = int(f.read(4)) # Length of vec embeddings in 

targets = set()
genre_targets = set()

genre_dict = dict()
embed_dict = dict()

# Fill target list
f_movies = open("IMDB_dataset/u.item", "r")
f_genre_names = open("IMDB_dataset/u.copy.genre", "r")

# Get all genre names
line = f_genre_names.readline()
while line:
    line = line.split("|")
    genre_targets.add(line[0].lower())
    line = f_genre_names.readline()

# Get all movie names
line = f_movies.readline()
while line:
    line = line.split("|")
    movie_title = line[1].split()
    filtered_title = list()
    for word in movie_title:
        word = word.lower()
        if "(" in word:
            continue
        word = ''.join(c for c in word if c.isalnum())
        targets.add(word)
            
    line = f_movies.readline()

# Begin das search
float_size = 4
for i in range(words):
    # Get word
    word = ""
    while True:
        letter = f.read(1)

        # If we get error, weird ass character, ignore
        try:
            if str(letter, 'utf8') == " ": break
        except:
            break
        word += str(letter, 'utf8')

    word = word.lower()

    # Read word embedding
    if word in targets:
        embed_vec = list()
        for j in range(embed_len):        
            embed_vec.append(struct.unpack('f', f.read(float_size)))
        
        embed_dict[word] = embed_vec
        targets.remove(word)
    
    elif word in genre_targets:
        embed_vec = list()
        for j in range(embed_len):        
            embed_vec.append(struct.unpack('f', f.read(float_size)))
        
        genre_dict[word] = embed_vec
    else:
        f.seek(f.tell() + (embed_len * float_size))

    if i % 100 == 0:
        print("{i}: {l}/{d}".format(i = i,l = len(embed_dict), d = len(targets)))

print("result: {i}/{d}".format(i = len(embed_dict), d = len(targets)))

f_pickle = open("IMDB_embed.pickle", "wb")
f_genre = open("genre_embed.pickle", "wb")

print(targets)
print(genre_dict.keys())
pickle.dump(embed_dict, f_pickle, protocol=pickle.HIGHEST_PROTOCOL)
pickle.dump(genre_dict, f_genre, protocol=pickle.HIGHEST_PROTOCOL)
    

    


