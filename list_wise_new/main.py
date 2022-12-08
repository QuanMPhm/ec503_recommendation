import itertools
import pandas as pd
import numpy as np
import random
import csv
import time

import matplotlib.pyplot as plt

import tensorflow as tf2
tf = tf2.compat.v1
tf.disable_v2_behavior()

import keras.backend as K
from keras import Sequential
from keras.layers import Dense, Dropout
from Environment import Environment
from Actor import Actor
from Critic import Critic
# from Data_Generator import DataGenerator
from Data_Generator_news import DataGenerator
from Embeddings import Embeddings
from keras.utils import plot_model

print('run main.py')

def read_file(data_path):

    data = pd.read_csv(data_path, sep=';')
    for col in ['state', 'n_state', 'action_reward']:
        data[col] = [np.array([[np.int(k) for k in ee.split('&')] for ee in e.split('|')]) for e in data[col]]
    for col in ['state', 'n_state']:
        data[col] = [np.array([e[0] for e in l]) for l in data[col]]

    data['action'] = [[e[0] for e in l] for l in data['action_reward']]
    data['reward'] = [tuple(e[1] for e in l) for l in data['action_reward']]
    data.drop(columns=['action_reward'], inplace=True)

    return data


def read_embeddings(embeddings_path):
    ''' Load embeddings (a vector for each item). '''

    embeddings = pd.read_csv(embeddings_path, sep=';')
    print(np.array([[np.float64(k) for k in e.split('|')]
                     for e in embeddings.iloc[:,1]]))
    return np.array([[np.float64(k) for k in e.split('|')]
                     for e in embeddings.iloc[:,1]])


def experience_replay(replay_memory, batch_size, actor, critic, embeddings, ra_length, state_space_size,
                      action_space_size, discount_factor):
    '''
    Experience replay.
    Args:
      replay_memory: replay memory D in article.
      batch_size: sample size.
      actor: Actor network.
      critic: Critic network.
      embeddings: Embeddings object.
      state_space_size: dimension of states.
      action_space_size: dimensions of actions.
    Returns:
      Best Q-value, loss of Critic network for printing/recording purpose.
    '''

    # '22: Sample minibatch of N transitions (s, a, r, s′) from D'
    samples = replay_memory.sample_batch(batch_size)
    states = np.array([s[0] for s in samples])
    actions = np.array([s[1] for s in samples])
    rewards = np.array([s[2] for s in samples])
    n_states = np.array([s[3] for s in samples]).reshape(-1, state_space_size)

    # '23: Generate a′ by target Actor network according to Algorithm 2'
    n_actions = actor.get_recommendation_list(ra_length, states, embeddings, target=True).reshape(-1, action_space_size)

    # Calculate predicted Q′(s′, a′|θ^µ′) value
    target_Q_value = critic.predict_target(n_states, n_actions, [ra_length] * batch_size)

    # '24: Set y = r + γQ′(s′, a′|θ^µ′)'
    expected_rewards = rewards + discount_factor * target_Q_value

    # '25: Update Critic by minimizing (y − Q(s, a|θ^µ))²'
    critic_Q_value, critic_loss, _ = critic.train(states, actions, [ra_length] * batch_size, expected_rewards)

    # '26: Update the Actor using the sampled policy gradient'
    action_gradients = critic.get_action_gradients(states, n_actions, [ra_length] * batch_size)
    actor.train(states, [ra_length] * batch_size, action_gradients)

    # '27: Update the Critic target networks'
    critic.update_target_network()

    # '28: Update the Actor target network'
    actor.update_target_network()

    return np.amax(critic_Q_value), critic_loss


# Hyperparameters
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

dg = DataGenerator('ml-100k/u.data', 'ml-100k/u.item')
dg.gen_train_test(0.8, seed=42)

dg.write_csv('train.csv', dg.train, nb_states=[history_length], nb_actions=[ra_length])
dg.write_csv('test.csv', dg.test, nb_states=[history_length], nb_actions=[ra_length])

data = read_file('train.csv')

# embeddings
embeddings = Embeddings(read_embeddings('embeddings.csv'))
state_space_size = embeddings.size() * history_length
action_space_size = embeddings.size() * ra_length



environment = Environment(data, embeddings, alpha, gamma, fixed_length)

tf.reset_default_graph() # For multiple consecutive executions

sess = tf.Session()
# '1: Initialize actor network f_θ^π and critic network Q(s, a|θ^µ) with random weights'
actor = Actor(sess, state_space_size, action_space_size, batch_size, ra_length, history_length, embeddings.size(), tau, actor_lr)
critic = Critic(sess, state_space_size, action_space_size, history_length, embeddings.size(), tau, critic_lr)




dict_embeddings = {}
for i, item in enumerate(embeddings.get_embedding_vector()):
  str_item = str(item)
  assert(str_item not in dict_embeddings)
  dict_embeddings[str_item] = i


def state_to_items(state, actor, ra_length, embeddings, dict_embeddings, target=False):
  return [dict_embeddings[str(action)]
          for action in
          actor.get_recommendation_list(ra_length, np.array(state).reshape(1, -1), embeddings, target).reshape(
              ra_length, embeddings.size())]

txt_writer = open('state',mode='w')

def test_actor(actor, test_df, embeddings, dict_embeddings, ra_length, history_length, target=False, nb_rounds=1):
  ratings = []
  unknown = 0
  random_seen = []
  for _ in range(nb_rounds):
      for i in range(len(test_df)):
          history_sample = list(test_df[i].sample(history_length)['itemId'])
          recommendation = state_to_items(embeddings.embed(history_sample), actor, ra_length, embeddings,
                                          dict_embeddings, target)
          for item in recommendation:
              l = list(test_df[i].loc[test_df[i]['itemId'] == item]['rating'])
              assert (len(l) < 2)
              if len(l) == 0:
                  unknown += 1
              else:
                  ratings.append(l[0])
          for item in history_sample:
              random_seen.append(list(test_df[i].loc[test_df[i]['itemId'] == item]['rating'])[0])

  return ratings, unknown, random_seen


saver = tf.train.Saver()
saver.restore(sess, 'save/' + 'listwise' + '.ckpt')



# plot_model(actor)
# plot_model(critic)
ratings, unknown, random_seen = test_actor(actor, dg.train, embeddings, dict_embeddings, ra_length, history_length, target=False, nb_rounds=10)
print('%0.1f%% unknown' % (100 * unknown / (len(ratings) + unknown)))
plt.figure()
# plt.subplot(1, 2, 1)
plt.title('recommend result')
plt.hist(ratings)
plt.title('Predictions ; Mean = %.4f' % (np.mean(ratings)))
plt.xlabel('ratings on recommended items')
plt.ylabel('number of items')
# plt.subplot(1, 2, 2)
# plt.hist(random_seen)
# plt.title('Random ; Mean = %.4f' % (np.mean(random_seen)))
plt.show()

# if __name__ == '__main__':
#     pass

