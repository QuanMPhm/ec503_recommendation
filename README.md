My role was handling state representation and preliminary data visualization. This was done for the reinforcement learning (RL) project which used the MovieLens dataset
The following files and folders should be the ones with the most interest:
- data_analysis.ipynb: Contains python code to perform some basic visualization and analysis of the dataset
- Data_Analysis_Plots: Contains the matplotlib plots outputed by data_analysis.ipynb
- data_representation.py: Contains several function which implemented different methods of state representation for the step transition function, to be used to train the RL algorhythm
- embed_extract.py: Contains code which will extract the word2vec embeddings of the movie titles and genres from Google's public word2vec dataset. References are contained in that file
- IMDB_dataset: Contains the MovieLens dataset