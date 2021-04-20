import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from Sample import Sample
import matplotlib.pyplot as plt
from tensorflow.keras import optimizers

keras.backend.set_floatx('float64')
class Actor():
    def __init__(self, env, n_action,  action_bound, trainable = True):
        self.n_actions = int(n_action)
        self.action_bound = action_bound
        self.env = env
        self.n_features = self.env.observation_space.shape[0]
        self.a_f1 = layers.Dense(units= 100, activation='relu', trainable=trainable)
        self.a_mu = layers.Dense(units=self.n_actions, activation='tanh', trainable=trainable)
        self.a_sigma =layers.Dense(units=self.n_actions, activation='softplus', trainable=trainable)

    def build_net(self, input_shape):
        inputs = keras.Input(shape=input_shape)
        x = self.a_f1(inputs)
        a_mu = self.a_mu(x) * 2
        a_sigma = self.a_sigma(x)
        # normal_dist =

        model = keras.Model(inputs = inputs, outputs = (a_mu, a_sigma))
        return model


    def choose_action(self,state):
        actor_net = self.build_net(state.shape)

        (a_mu, a_sigma) = actor_net(state)
        pi_a = tfp.distributions.Normal(a_mu,a_sigma)
        action = tf.squeeze(pi_a.sample(1), 0)
        # pi_action = [tf.random.normal([1],mean = amu[i], stddev = a_sigma[i])[0].numpy() for i in range(self.n_actions)]
        action = tf.clip_by_value(action, self.action_bound[0], self.action_bound[1])
        action = tf.squeeze(action ,axis = 0)

        return action.numpy(), tf.squeeze(pi_a.prob(action.numpy()) , 0).numpy() #shape: 动作的维度

class Critic():
    def __init__(self, trainable = True):


        self.c_f1 = layers.Dense(units= 100, activation='relu', trainable=trainable)
        self.v = layers.Dense(1,trainable = trainable)

    def build_net(self, input_shape):
        inputs = keras.Input(shape=input_shape)
        x = self.c_f1(inputs)
        value = self.v(x)
        model = keras.Model(inputs = inputs, outputs = value)
        return model

    def get_value(self,state):
        critic_net = self.build_net(state.shape)
        value = critic_net(state)
        if value.shape[0] == 1:
            value = tf.squeeze(value, 0)

        return value


class PPO():
    def __init__(self, Actor, Critic, env, lr = 0.001, n_actions = 1,  model_file = None):
        self.env = env
        self.lr = lr
        self.actor = Actor
        self.critic = Critic
        self.critic_train = self.critic(trainable = True)
        self.action_bound = [-env.action_space.high, env.action_space.high]
        self.n_features = env.observation_space.shape[0]
        self.n_actions = n_actions
        self.actor_train = self.actor(env, n_actions, self.action_bound, trainable = True)
        self.actor_old = self.actor(env, n_actions, self.action_bound, trainable = False)
        self.actor_net = self.actor_train.build_net(input_shape=self.n_features)
        self.critic_net = self.critic_train.build_net(input_shape= self.n_features)
        self.actor_oldnet= self.actor_old.build_net(input_shape=self.n_features)
        # self.pi = None
        # self.pi_old = None
        self.Sample = Sample(env, self.actor_train, self.critic_train)
    def update_old_pi(self):


        self.actor_oldnet.set_weights(self.actor_net.get_weights())

    def current_pi(self, actor_net,state):
        (amu,a_sigma) = actor_net(state)

        return tfp.distributions.Normal(amu,a_sigma)

    def compute_loss(self,state,action,reward ,epsilon = 0.2):
        current_pi = self.current_pi(self.actor_net, state)
        old_pi = self.current_pi(self.actor_oldnet, state)
        pi_a = current_pi.prob(action)
        pi_old = old_pi.prob(action)
        ratio = pi_a / pi_old
        # print("ratio is ", ratio)
        adv = reward - self.critic_train.get_value(state)
        c_loss = tf.reduce_mean(tf.square(adv))
        adv = np.reshape(adv, [len(adv),1])
        surr = ratio * adv

        a_loss = -tf.reduce_mean(tf.minimum(surr, tf.clip_by_value(ratio, 1 -epsilon, 1+epsilon) * adv))
        # c_loss = tf.reduce_mean(tf.square(adv))
        # adv = tf.convert_to_tensor(adv, tf.float32)
        # c_loss = adv
        return a_loss, c_loss

    def test_policy(self,env, RENDER, total_test = 1):
        reward_sum = []
        for i in range(total_test):
            state = env.reset()
            if RENDER:
                print('Test %d, init: %f, %f, %f' %(i+1,state[0], state[1], state[2]))

            while True:
                reward_sum_one = 0
                if RENDER:
                    env.render()
                state = np.reshape(state, [1,3])
                action = self.actor_train.choose_action(state)
                state_, reward, done, info = env.step(action)
                reward_sum_one +=reward
                if done:
                    if RENDER:
                        print('the total reward of Test %d: %f'%(i+1, reward_sum_one))
                    break
                state = state_
            reward_sum.append(reward_sum_one)
        return  reward_sum

    def train(self, episode):
        reward_sum = 0
        average_reward_line = []
        training_time = []
        average_reward = 0
        current_total_reward = 0
        optimizer = optimizers.Adam(self.lr)
        TOTAL_update = 10
        for i in range(episode):
            current_state, current_action, current_value = self.Sample.sample_episodes(1)
            #old net 采样出一条链数据后续进行多次使用
            self.update_old_pi()
            # print(self.actor_net.get_weights()[0] == self.actor_oldnet.get_weights()[0])
            for _ in range(TOTAL_update):
                with tf.GradientTape(persistent=True) as tape:
                    a_loss, c_loss = self.compute_loss(current_state, current_action, current_value)
                    # loss = a_loss + c_loss
                grads_c = tape.gradient(c_loss, self.critic_net.trainable_variables)
                grads_a = tape.gradient(a_loss, self.actor_net.trainable_variables)


                # print(self.critic_net.trainable_variables)
                optimizer.apply_gradients(zip(grads_a, self.actor_net.trainable_variables))
                optimizer.apply_gradients(zip(grads_c, self.critic_net.trainable_variables))
                # print(_)
            # print(self.actor_net.get_weights()[0] == self.actor_oldnet.get_weights()[0])
            current_total_reward = self.test_policy(self.env, True)[0]
            if i ==0:
                average_reward = current_total_reward
            else:
                average_reward = 0.95 * average_reward + 0.05*current_total_reward
            average_reward_line.append(average_reward)
            training_time.append(i)
            # if average_reward >-300:
            #     break

            print('current_reward is %f '%(average_reward))
        self.plot_result(training_time, average_reward_line)

    def plot_result(self, time, average_reward):
        plt.plot(time,average_reward)
        plt.xlabel("training step")
        plt.ylabel("score")
        plt.show()

