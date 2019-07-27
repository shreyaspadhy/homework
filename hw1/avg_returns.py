import os
import os.path as op
import pickle
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import numpy as np
import gym
import load_policy
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import models
from utils import *
import glob
import re


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--num_train_rolls', type=int, default=20)
    parser.add_argument('--num_its', type=int, default=10)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--train', action='store_true')
    args = parser.parse_args()

    env = gym.make(args.envname)
    max_steps = env.spec.max_episode_steps
    policy_fn = load_policy.load_policy(args.expert_policy_file)

    policies = {}
    policies['expert'] = policy_fn
    # Store checkpoints as well
    policies['BC'] = models.Net(obs_size=env.observation_space.shape[0],
                                hidden=256,
                                act_space_size=env.action_space.shape[0])

    policies['Dagger'] = models.Net(obs_size=env.observation_space.shape[0],
                                    hidden=256,
                                    act_space_size=env.action_space.shape[0])

    policy_wts = {}

    regex = re.compile('.*{}_iters'.format(args.num_its - 1))
    for root, dirs, files in sorted(os.walk('./checkpoints/Dagger/{}'.format(args.envname))):
        for dir in dirs:
            if regex.match(dir):
                policy_wts['Dagger'] = op.join(root, dir, 'cp.ckpt')

    regex = re.compile('.*{}_train_rolls'.format(args.num_train_rolls))
    for root, dirs, files in sorted(os.walk('./checkpoints/BC/{}'.format(args.envname))):
        for dir in dirs:
            if regex.match(dir):
                policy_wts['BC'] = op.join(root, dir, 'cp.ckpt')

    print(policy_wts['Dagger'], policy_wts['BC'])
    checkpoint_path = policy_wts['Dagger']
    checkpoint_dir = os.path.dirname(checkpoint_path)
    restore_checkpoint = tf.train.Checkpoint(model=policies['Dagger'])
    restore_checkpoint.restore(checkpoint_path).expect_partial()

    checkpoint_path = policy_wts['BC']
    checkpoint_dir = os.path.dirname(checkpoint_path)
    restore_checkpoint = tf.train.Checkpoint(model=policies['BC'])
    restore_checkpoint.restore(checkpoint_path).expect_partial()

    # Visualize the trained policy and calculate returns
    returns, avg_returns = {}, {}
    num_episodes = 20

    for policy in ['expert', 'BC', 'Dagger']:
        returns[policy] = []
        for i in range(num_episodes):
            returns[policy].append(run_policy(env, policies[policy], policy=policy))

        avg_returns[policy] = np.mean(returns[policy])

    # with open('{}_{}_eps_{}_epochs.pkl', 'wb') as file:
    #     pickle.dump(avg_returns, file, protocol=2)

        print("Average return from {} : ".format(policy), avg_returns[policy])

    # fig = plt.figure()
    # x_eps = np.arange(0, num_episodes)

    # sns.tsplot(time=x_eps, data=returns_BC, color='g', linestyle='-', label='Behaviour Cloning')
    # sns.tsplot(time=x_eps, data=returns, color='r', linestyle='-', label='Expert')

    # plt.ylabel("Returns per episode", fontsize=25)
    # plt.xlabel("Number of episodes", fontsize=25)
    # plt.title("Behaviour Cloning", fontsize=30)
    # plt.legend(loc='center right')
    # plt.show()


if __name__ == '__main__':
    main()
