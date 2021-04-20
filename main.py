from Net  import Actor, Critic,PPO
from Sample import Sample
import gym
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt
# import Config.Env as Env


def main():
    TOTAL = 1000

    env_name = 'Pendulum-v0'
    env = gym.make(env_name)
    env.unwrapped
    env.seed(1)
    action_bound = [-env.action_space.high, env.action_space.high]
    ppo1 = PPO(Actor,Critic,env,)
    ppo1.train(TOTAL)

if __name__ == "__main__":
    main()