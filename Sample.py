import tensorflow as tf
import numpy as np
import gym
import matplotlib.pyplot as plt
import time

class Sample():
    def __init__(self, env, policy_net, value_net, gamma = 0.9):
        self.env = env
        self.gamma = gamma
        self.actor = policy_net
        self.critic = value_net

    def sample_episodes(self, episodes, batch = 200, mini_batch =16):
        """

        :param episodes:产生的n条轨迹
        :batch: 即存储的一个大容器包含采样值数量
        :mini_batch: 采样一条链的状态数
        :return:
        """
        batch_states = []
        batch_values = []
        batch_actions = []
        for i in range(episodes):
            ob = self.env.reset()
            reward_episode = []
            mini_batch_states = []
            mini_batch_actions = []
            mini_batch_rewards = []
            minibatch_items_num = 0 # TD(n)
            batch_items_num = 0
            for j in range(batch):
                flag = 1
                state = tf.reshape(ob,[1,3])
                action, pi_a = self.actor.choose_action(state)
                action = action.tolist()
                ob_, reward, done, info = self.env.step(action)
                state_ = tf.reshape(ob_,[1,3])
                mini_batch_states.append(state)
                mini_batch_actions.append(action)
                mini_batch_rewards.append(reward)
                minibatch_items_num +=1
                batch_items_num = j + 1
                if minibatch_items_num == mini_batch or batch_items_num == batch:
                    #到达mini_batch链的最后一个状态
                    value_last = self.critic.get_value(state_)#最后一个状态的估计V值
                    #反向传递 v(t) = gamma * v(t+1) + r
                    value = value_last
                    values = np.zeros_like(mini_batch_rewards) #存储每一个状态的v值

                    for t in reversed(range(0, len(mini_batch_rewards))):
                        value = value * self.gamma + mini_batch_rewards[t]
                        values[t] = value

                    for t in range(len(mini_batch_rewards)):
                        batch_values.append(values[t])
                        batch_states.append(mini_batch_states[t])
                        batch_actions.append(mini_batch_actions[t])

                    minibatch_items_num = 0

                    mini_batch_states = []
                    mini_batch_actions = []
                    mini_batch_rewards = []
                ob = ob_
        # batch_actions = batch_actions.tolist()
        batch_states = np.reshape(batch_states, [len(batch_states), self.actor.n_features])
        batch_actions = np.reshape(batch_actions, [len(batch_actions), 1])
        batch_values = np.reshape(batch_values, [len(batch_values), 1])

        return batch_states , batch_actions , batch_values
