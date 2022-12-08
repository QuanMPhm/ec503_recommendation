from dustbin.produce_dataset import generateSet
from RL_brain import DeepQNetwork

import numpy as np
import tensorflow as tf2
tf = tf2.compat.v1
tf.disable_v2_behavior()

from run_me import DQN_test
import matplotlib.pyplot as plt
import random

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings("ignore")

# todo: change parameters below
numOfMAB=10
number_he_like = 2
#############
train_set_size = 10000
tf.reset_default_graph()

iter_batch_size = 50
train_set_batch_size = 250
testset_size = 250
episode_size = 20
epsilon = 0.15  # if the preference of the predicted choice is around the range of 0.15 of the best one, take it as a good prediction
max_ccr = 0

def cal_CCR(episode):
    ccr_train[iter_batch*20+episode] = start_testing_train()
    ccr_test[iter_batch * 20 + episode] = start_testing_test()
    print('ccr_train', iter_batch*20+episode, ' = ', ccr_train[iter_batch*20+episode])

    print('ccr_test', iter_batch * 20 + episode, ' = ', ccr_test[iter_batch * 20 + episode])

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

def start_testing_test():
    global max_ccr
    ccr = 0
    s = np.zeros(numOfMAB)
    for i in range(testset_size):
        for j in range(numOfMAB):
            s[j] = np.random.rand()
        s = np.array(s).flatten()
        action,actions_value = RL.choose_action(s)
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
        # initial observation
        chosen_person = np.random.randint(low=1,high=train_set_size,size=1)
        # print(chosen_person)
        # print(env.preference[chosen_person,:])
        s = env.preference[chosen_person,:]
        s = np.array(s).flatten()
        for i in range(300):
            # fresh env
            # todo

            # RL choose action based on observation
            action,actions_value = RL.choose_action(s)

            # RL take action and get next observation and reward
            # todo: this is the original reward
            # reward = env.step(s=chosen_person,action=action,actions_value=actions_value)

            reward,cosine_similarity = env.step(person_choose=chosen_person,s=env.preference[chosen_person,:],action=action,actions_value=actions_value)

            RL.store_transition(s, action, reward, s_ = np.array(env.preference[chosen_person,:]).flatten() )

            if (step > 0) and (step % 5 == 0):
                RL.learn()

            # swap observation
            #observation = s_
            step += 1
        cal_CCR(episode)

def plot_CCR():
    #plot the testing CCR
    ccr_x = range(iter_batch_size*episode_size)
    plt.figure()
    plt.plot(ccr_x,ccr_train)
    plt.xlabel('iteration')
    plt.ylabel('CCR')
    plt.title('traing ccr action space = 100')
    plt.show()
    #plot the training CCR
    ccr_x = range(iter_batch_size*episode_size)
    plt.figure()
    plt.plot(ccr_x,ccr_train)
    plt.xlabel('iteration')
    plt.ylabel('CCR')
    plt.title('testing ccr action space = 100')
    plt.show()

if __name__ == '__main__':
    env = generateSet(numOfMAB=numOfMAB,number_he_like=number_he_like)
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
    #ccr40 = np.zeros(800)
    for iter_batch in range(iter_batch_size):
        print('num of iter: ', iter_batch)
        run_recommend(iter_batch,env,RL)

        #test_agent.start_testing()

        test_s = random.randint(0, train_set_size)
        print('%%%%%%%%%%%%%%%%%%%%%%%train')
        print(test_s)
        print('the preference of the chosen person for ten news types: ', env.preference[test_s, :])
        s=env.preference[test_s, :]
        s = np.array(s).flatten()
        print('highest preference: ',RL.choose_action(s))
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
