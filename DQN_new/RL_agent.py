import numpy as np
import tensorflow as tf2
tf = tf2.compat.v1
tf.disable_v2_behavior()

class DQN:
    def __init__(self,num_action=10,
                 s_features=10,
                 learning_rate=0.01,
                 reward_decay=0.9,
                 e_greedy=0.9,
                 e_greedy_increment=None,
                 output_graph=False,
                 memory_size=500,
                 batch_size=32,
                 replace_target_iter=100):
        self.num_action = num_action
        self.s_features = s_features
        self.learning_rate = learning_rate
        self.reward_decay = reward_decay
        self.e_greedy = e_greedy
        self.e_greedy_increment = e_greedy_increment
        self.output_graph = output_graph
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon = 0 if e_greedy_increment is not None else self.e_greedy
        self.sess = tf.Session()
        #renew the target net
        self.learn_step_counter = 0
        self.replace_target_iter = replace_target_iter

        #store the weight and bias
        self.init_weight_bias()

        if self.output_graph == True:
            print(self.sess.graph)

    def init_net(self):
        self.s = tf.placeholder(tf.float32, [None, self.s_features], name='s')  # input
        self.q_target = tf.placeholder(tf.float32, [None, self.num_action], name='Q_target')  # for calculating loss
        self.s_ = tf.placeholder(tf.float32, [None, self.s_features], name='s_')  # input

        with tf.variable_scope('pred_net'):

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                l1 = self.MLP(input_tensor=self.s,w=self.w1,b=self.b1)
                l1 = self.Relu(l1)

            with tf.variable_scope('l2'):
                self.q_eval = self.MLP(input_tensor=l1,w=self.w2,b=self.b2)

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)

        with tf.variable_scope('target_net'):
            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                l1 = self.MLP(input_tensor=self.s,w=self.w1t,b=self.b1t)
                l1 = self.Relu(l1)

            with tf.variable_scope('l2'):
                self.q_target = self.MLP(input_tensor=l1,w=self.w2t,b=self.b2t)

    def init_weight_bias(self):
        n_l1, w_initializer, b_initializer = 10, \
                                             tf.random_normal_initializer(0., 0.3), tf.constant_initializer(
            0.1)  # config of layers
        with tf.variable_scope('l1'):
            self.w1 = tf.Variable('w1', [self.s_features, n_l1], initializer=w_initializer)
            self.b1 = tf.Variable('b1', [1, n_l1], initializer=b_initializer)
        with tf.variable_scope('l1'):
            self.w2 = tf.Variable('w2', [self.s_features, n_l1], initializer=w_initializer)
            self.b2 = tf.Variable('b2', [1, n_l1], initializer=b_initializer)
        with tf.variable_scope('l1_target'):
            self.w1t = tf.Variable('w1t', [self.s_features, n_l1], initializer=w_initializer)
            self.b1t = tf.Variable('b1t', [1, n_l1], initializer=b_initializer)
        with tf.variable_scope('l1'):
            self.w2t = tf.Variable('w2t', [self.s_features, n_l1], initializer=w_initializer)
            self.b2t = tf.Variable('b2t', [1, n_l1], initializer=b_initializer)

    def choose_action(self, s):
        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: s})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.num_action)
        return action

    def learn(self):
        self.renew_the_target_net()
        self.learn_step_counter += 1


    def MLP(self,input_tensor,w,b):
        output = tf.matmul(input_tensor, w) + b
        return output

    def Relu(self,input_tensor):
        output = tf.nn.relu(input_tensor)
        return output

    def renew_the_target_net(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.replace_target_op = [tf.assign(t, e) for t, e in zip([self.w1t, self.b1t, self.w2t, self.b2t],
                                                                      [self.w1, self.b1, self.w2, self.b2])]
            self.sess.run(self.replace_target_op)
            print('\ntarget_params_replaced\n')