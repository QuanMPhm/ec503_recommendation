from produce_dataset import generateSet
from RL_brain import DeepQNetwork
# from RL_brain_test import DeepQNetwork_test

import numpy as np
import tensorflow as tf2
tf = tf2.compat.v1
tf.disable_v2_behavior()

# from run_me import DQN_test
import matplotlib.pyplot as plt
import random

# ???
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings("ignore")

# Number of people?
train_set_size = 10000
tf.reset_default_graph()

train_set_batch_size = 800
testset_size = 800
epsilon = 0.15  # if the preference of the predicted choice is around the range of 0.15 of the best one, take it as a good prediction


def cal_CCR(episode):
    ccr_train[iter_batch*20+episode] = start_testing_train()
    ccr_test[iter_batch * 20 + episode] = start_testing_test()
    print('ccr_train', iter_batch*20+episode, ' = ', ccr_train[iter_batch*20+episode])

    print('ccr_test', iter_batch * 20 + episode, ' = ', ccr_test[iter_batch * 20 + episode])

## Pick 800 people from dataset to test
def start_testing_train():
    ccr = 0
    for i in range(train_set_batch_size):
        person_choose = np.random.randint(0, train_set_size, size=1)
        s = env.preference[person_choose, :]
        s = np.array(s).flatten()
        action = RL.choose_action(s)
        if (max(s) - s[action]) <= epsilon:
            ccr += 1
    return ccr/train_set_batch_size

## Generate 800 random new people to test...
def start_testing_test():
    ccr = 0
    s = np.zeros(10)
    for i in range(testset_size):
        for j in range(10):
            s[j] = np.random.rand()
        s = np.array(s).flatten()
        action = RL.choose_action(s)
        if (max(s) - s[action]) <= epsilon:
            ccr += 1
    return ccr/testset_size

def run_recommend(iter_batch,env,RL):
    step = 0

    # For each episode, chose 20 people
    for episode in range(20):
        # initial observation
        chosen_person = np.random.randint(low=1,high=train_set_size,size=1)
        # print(chosen_person)
        # print(env.preference[chosen_person,:])

        ## ???
        s = env.preference[chosen_person,:]
        s = np.array(s).flatten()
        
        for i in range(300):
            # fresh env
            # todo

            # RL choose action based on observation
            # ?? What is RL
            action = RL.choose_action(s)

            # RL take action and get next observation and reward
            reward = env.step(s=chosen_person,action=action)

            RL.store_transition(s, action, reward, s_ = np.array(env.preference[chosen_person,:]).flatten() )

            # Learn every 5 reconmendations
            if (step > 0) and (step % 5 == 0):
                RL.learn()

            # swap observation
            #observation = s_
            step += 1
        cal_CCR(episode)

def plot_CCR():
    #plot the training CCR
    ccr_x = range(800)
    plt.figure()
    plt.plot(ccr_x,ccr_train)
    plt.xlabel('iteration')
    plt.ylabel('CCR')
    plt.title('traing ccr')
    plt.show()
    #plot the training CCR
    ccr_x = range(800)
    plt.figure()
    plt.plot(ccr_x,ccr_train)
    plt.xlabel('iteration')
    plt.ylabel('CCR')
    plt.title('testing ccr')
    plt.show()

if __name__ == '__main__':
    env = generateSet() # ??
    # test_agent = DQN_test() # ??
    tf.reset_default_graph() # ??
    RL = DeepQNetwork(env.MAB, env.MAB,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=300,
                      memory_size=2000,
                      # output_graph=True
                      ) # Initialize learner??

    ccr_test = np.zeros(800)
    ccr_train = np.zeros(800)
    ccr40 = np.zeros(800) # ??

    ## 20 people for 10 iterations = 200 people??
    for iter_batch in range(10):
        print('num of iter: ',iter_batch)

        ## Does the RL model reset each episode??, each time ti learns?
        ## Train and learn with 200 people
        run_recommend(iter_batch,env,RL) # ??

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

        # ??
        RL.savenet()
        #ccr[iter_batch] = test_agent.start_testing()
        # print('!!!!!!!!!!!!!!!!!ccr800', iter_batch, ' = ', ccr[iter_batch])
        #print(test)
    x = 1
    print('model saved')
    plot_CCR()

