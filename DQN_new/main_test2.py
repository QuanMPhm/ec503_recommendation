from produce_dataset_cosine import generateSet
from RL_brain_test import DeepQNetwork_test
from run_me import DQN_test
import math
import numpy as np
import tensorflow as tf2
tf = tf2.compat.v1
tf.disable_v2_behavior()

import matplotlib.pyplot as plt
import random

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings("ignore")

numOfPerson = 10000
# todo: change parameters below
numOfMAB=10
number_he_like = 2
train_title = 'traing ccr action space = 10'
test_title = 'testing ccr action space = 10'
cosine_simi_title = 'cosine similarity actionspace = 10'
iter_batch_size = 120
more_layer_num = 3
#more_layer_num = 5
from RL_brain_cosine import DeepQNetwork
# from RL_brain_cosine_more_neurons import DeepQNetwork
# from RL_brain_cosine_more_layer import DeepQNetwork
# init_method = 'random'
init_method='like&hate'
# init_method='like&ok'
# reward_methods='more than'
# reward_methods='cosine_linear'
reward_methods='cosine_exp'

# todo: end
#############
train_set_size = 10000
test_set_size = 10000
tf.reset_default_graph()

train_set_batch_size = 250
testset_size = 250
episode_size = 20
epsilon = 0.15  # if the preference of the predicted choice is around the range of 0.15 of the best one, take it as a good prediction
max_ccr = 0

def def_test_preference():
    global test_preference
    test_preference = np.zeros(numOfPerson, numOfMAB)
    if init_method == 'like&hate':
        for i in range(numOfPerson):
            index_like = np.random.randint(1, env.MAB, [1, number_he_like])
            for j in range(number_he_like):
                test_preference[i, index_like[0][j]] = np.random.randint(math.floor(0.9 * env.MAB * 10),
                                                                         env.MAB * 10) / (env.MAB * 10)
    elif init_method == 'random':
        for i in range(numOfPerson):
            for j in range(env.MAB):
                test_preference[i, j] = np.random.rand()
    else:
        print('def_test_preference error')

def cal_CCR(episode,cosine_similarity,actions_value,s):
    ccr_train[iter_batch*20+episode] = start_testing_train()
    ccr_test[iter_batch * 20 + episode] = start_testing_test(test_preference=test_preference)
    cosine_similarity =  np.float(cosine_similarity)
    cosine_similarity_store[iter_batch * 20 + episode] = cosine_similarity
    # print('cosine_similarity_store[iter_batch * 20 + episode]',cosine_similarity_store[iter_batch * 20 + episode])
    print('cosine_similarity',cosine_similarity)
    print('ccr_train', iter_batch*20+episode, ' = ', ccr_train[iter_batch*20+episode])
    print('ccr_test', iter_batch * 20 + episode, ' = ', ccr_test[iter_batch * 20 + episode])
    print('type cs = ',cosine_similarity)
    print('actions_value = ',actions_value)
    print('s = ',s)
    print('#####################')


def start_testing_train():
    ccr = 0
    for i in range(train_set_batch_size):
        person_choose = np.random.randint(0, train_set_size, size=1)
        s = env.preference[person_choose, :]
        s = np.array(s).flatten()
        action,actions_value = RL.choose_action(s)
        if (max(s) - s[action]) <= epsilon:
            ccr += 1

    return ccr/train_set_batch_size

def start_testing_test(test_preference):
    global max_ccr
    ccr = 0
    s = np.zeros(numOfMAB)
    if init_method == 'restart_init':
        test_preference = np.zeros(numOfPerson, numOfMAB)
        for i in range(numOfPerson):
            index_like = np.random.randint(1, env.MAB, [1, number_he_like])
            for j in range(number_he_like):
                test_preference[i, index_like[0][j]] = np.random.randint(math.floor(0.9 * env.MAB * 10),
                                                                         env.MAB * 10) / (env.MAB * 10)
    else:
        for i in range(testset_size):
            person_choose = np.random.randint(0, test_set_size, size=1)
            s = test_preference[person_choose, :]
            s = np.array(s).flatten()
            action, actions_value = RL.choose_action(s)
            if (max(s) - s[action]) <= epsilon:
                ccr += 1
    # renew the model
    if ccr/testset_size > max_ccr:
        RL.savenet()
        print('model saved')
        max_ccr = ccr/testset_size
    return ccr/testset_size

def run_recommend(iter_batch,env,RL):
    step = 0
    for episode in range(episode_size):

        for i in range(300):
            # initial observation
            chosen_person = np.random.randint(low=1, high=train_set_size, size=1)
            # print(chosen_person)
            # print(env.preference[chosen_person,:])
            s = env.preference[chosen_person, :]
            s = np.array(s).flatten()


            # fresh env
            # todo

            # RL choose action based on observation
            action,actions_value = RL.choose_action(s)

            # RL take action and get next observation and reward
            # todo: this is the original reward
            # reward = env.step(s=chosen_person,action=action,actions_value=actions_value)

            reward,cosine_similarity,actions_value = env.step(person_choose=chosen_person,s=env.preference[chosen_person,:],action=action,actions_value=actions_value)
            #print('cosine_similarity out:',cosine_similarity)
            RL.store_transition(s, action, reward, s_ = np.array(env.preference[chosen_person,:]).flatten() )

            if (step > 0) and (step % 5 == 0):
                RL.learn()

            # swap observation
            #observation = s_
            step += 1
        cal_CCR(episode,cosine_similarity,actions_value,s)
    return cosine_similarity


def plot_CCR():
    #plot the testing CCR
    ccr_x = range(iter_batch_size*episode_size)
    plt.figure()
    plt.plot(ccr_x,ccr_train)
    plt.xlabel('iteration')
    plt.ylabel('CCR')
    plt.title(train_title)
    plt.show()
    #plot the training CCR
    ccr_x = range(iter_batch_size*episode_size)
    plt.figure()
    plt.plot(ccr_x,ccr_test)
    plt.xlabel('iteration')
    plt.ylabel('CCR')
    plt.title(test_title)
    plt.show()
    ccr_x = range(iter_batch_size*episode_size)
    plt.figure()
    plt.plot(ccr_x,cosine_similarity_store)
    plt.xlabel('iteration')
    plt.ylabel('cosine similarity')
    plt.title(cosine_simi_title)
    plt.show()

if __name__ == '__main__':
    env = generateSet(numOfPerson= numOfPerson,numOfMAB=numOfMAB,number_he_like=number_he_like,method=init_method,reward_method=reward_methods)
    def_test_preference()
    test_agent = DQN_test()
    tf.reset_default_graph()
    RL = DeepQNetwork(env.MAB, env.MAB,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=300,
                      memory_size=2000,
                      # output_graph=True
                      )
    ccr_test = np.zeros(iter_batch_size*episode_size)
    ccr_train = np.zeros(iter_batch_size*episode_size)
    cosine_similarity_store = np.zeros(iter_batch_size*episode_size)
    #ccr40 = np.zeros(800)
    for iter_batch in range(iter_batch_size):
        print('num of iter: ', iter_batch)
        cosine_similarity = run_recommend(iter_batch,env,RL)

        #test_agent.start_testing()

        test_s = random.randint(0, train_set_size)
        print('%%%%%%%%%%%%%%%%%%%%%%%train')
        print(test_s)
        print('the preference of the chosen person for ten news types: ', env.preference[test_s, :])
        s=env.preference[test_s, :]
        s = np.array(s).flatten()
        my_action,_ = RL.choose_action(s)
        print('highest preference: ',my_action)
        print('cosine_simi:',cosine_similarity)
        print('%%%%%%%%%%%%%%%%%%%%%%%test')
        ss = np.array([.1,.2,.3,.4,.4,.3,.2,.1,.9,.1])
        test = ss.flatten()
        RL.savenet()
        #ccr[iter_batch] = test_agent.start_testing()
        # print('!!!!!!!!!!!!!!!!!ccr800', iter_batch, ' = ', ccr[iter_batch])
        #print(test)
    x = 1
    plot_CCR()

#tensorboard --logdir=logs
