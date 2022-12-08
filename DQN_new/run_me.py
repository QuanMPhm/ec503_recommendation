from RL_brain_test import DeepQNetwork_test

import tensorflow as tf2
tf = tf2.compat.v1
tf.disable_v2_behavior()
import numpy as np
from dustbin.produce_dataset import generateSet
testset_size = 1000
epsilon = 0.15  # if the preference of the predicted choice is around the range of 0.15 of the best one, take it as a good prediction
# env = generateSet()
# RL = DeepQNetwork(env.MAB, env.MAB,
#                   learning_rate=0.01,
#                   reward_decay=0.9,
#                   e_greedy=0.9,
#                   replace_target_iter=200,
#                   memory_size=2000,
#                   # output_graph=True
#                   )
class  DQN_test:
    def __init__(self):
        self.env = generateSet()
        self.RL =  DeepQNetwork_test(self.env.MAB, self.env.MAB,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000,
                      # output_graph=True
                      )

    def start_testing(self):
        self.RL.restore()
        s = np.zeros(10)
        ccr = 0
        for k in range(testset_size):
            np.random.seed(k)
            for i in range(10):
                s[i] = np.random.rand()

            action = self.RL.choose_action(s)
            if (max(s) - s[action]) <= epsilon:
                # print('xmax = ', max(s))
                # print('pred = ', s[action])
                ccr += 1
        return ccr/testset_size

# if __name__ == '__main__':
#     env = generateSet()
#     RL = DeepQNetwork(env.MAB, env.MAB,
#                       learning_rate=0.01,
#                       reward_decay=0.9,
#                       e_greedy=0.9,
#                       replace_target_iter=200,
#                       memory_size=2000,
#                       # output_graph=True
#                       )
#
#     model_file = f'save/file'
#     RL.restore()
#     # s = np.zeros(10)
#     # ccr = 0
#     # for k in range(testset_size):
#     #     np.random.seed(k)
#     #     for i in range(10):
#     #         s[i] = np.random.rand()
#     #
#     #     action = RL.choose_action(s)
#     #     if ( max(s) - s[action]) <= epsilon:
#     #         print('xmax = ',max(s))
#     #         print('pred = ',s[action])
#     #         ccr += 1
#     #     print('input state = ',s)
#     #     print('action = ',action)
#     #     print('#######################')
#     ccr = start_testing()
#     print('eposilon = ',epsilon)
#     print('ccr = ',ccr)
#     print('testset_size = ',testset_size)
