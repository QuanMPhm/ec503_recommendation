from EmbeddingsGenerator import EmbeddingsGenerator
from Data_Generator_news import DataGenerator
import numpy as np
import pandas as pd
import cpickle

print('run training_embeddings.py')
def read_file(data_path):
  ''' Load data from train.csv or test.csv. '''

  data = pd.read_csv(data_path, sep=';')
  for col in ['state', 'n_state', 'action_reward']:
    data[col] = [np.array([[np.int(k) for k in ee.split('&')] for ee in e.split('|')]) for e in data[col]]
  for col in ['state', 'n_state']:
    data[col] = [np.array([e[0] for e in l]) for l in data[col]]

  data['action'] = [[e[0] for e in l] for l in data['action_reward']]
  data['reward'] = [tuple(e[1] for e in l) for l in data['action_reward']]
  data.drop(columns=['action_reward'], inplace=True)

  return data

# Hyperparameters
history_length = 12 # N in article
ra_length = 4 # K in article
discount_factor = 0.99 # Gamma in Bellman equation
actor_lr = 0.0001
critic_lr = 0.001
tau = 0.001 # τ in Algorithm 3
batch_size = 64
nb_episodes = 100
nb_rounds = 50
filename_summary = 'summary.txt'
alpha = 0.5 # α (alpha) in Equation (1)
gamma = 0.9 # Γ (Gamma) in Equation (4)
buffer_size = 1000000 # Size of replay memory D in article
fixed_length = True # Fixed memory length

process_DG = True
if process_DG == True:
    dg = DataGenerator('Preprocess_MIND_small/csv_trans','Preprocess_MIND_small/item_csv')
    dg.gen_train_test(0.8, seed=42)
    dg.save_variables()
else:
    dg = cpickle.load('DG_data_news/DG')

# dg.write_csv('Embedding_news/train_news.csv', dg.train, nb_states=[history_length], nb_actions=[ra_length])
# dg.write_csv('Embedding_news/test_news.csv', dg.test, nb_states=[history_length], nb_actions=[ra_length])
#
# data = read_file('train_news.csv')

if True: # Generate embeddings?
    eg = EmbeddingsGenerator(dg.user_train, pd.read_csv('Preprocess_MIND_small/csv_trans', sep=' ', names=['userId', 'itemId', 'rating', 'timestamp']))
    eg.train(nb_epochs=300)
    train_loss, train_accuracy = eg.test(dg.user_train)
    print('Train set: Loss=%.4f ; Accuracy=%.1f%%' % (train_loss, train_accuracy * 100))
    test_loss, test_accuracy = eg.test(dg.user_test)
    print('Test set: Loss=%.4f ; Accuracy=%.1f%%' % (test_loss, test_accuracy * 100))
    eg.save_embeddings('embeddings_news.csv')