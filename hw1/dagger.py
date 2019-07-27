#!/usr/bin/env python

"""
Code for DAgger
Example usage:
    python dagger.py ./expert_data/Humanoid-v2.pkl Humanoid-v2.pkl --num_train_rolls 20 --num_its 10 --train

Author of this script: Shreyas Padhy (shreyaspadhy@gmail.com)
"""

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
import models
from utils import *


def train_dagger(env, expert, policy, num_its=10, num_epochs=5, data={},
                 cp_dir='checkpoints/Dagger', cp_name=''):
    X_train_D, y_train_D = data['X']['train'].copy(), data['y']['train'].copy()

    # Create checkpoint callback
    for it in range(num_its):
        checkpoint_path = "{}/{}/{}{}_iters/cp.ckpt".format(cp_dir, env.unwrapped.spec.id, cp_name, it)
        checkpoint_dir = os.path.dirname(checkpoint_path)
        os.makedirs(checkpoint_dir, exist_ok=True)

        cp_callback = keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                      save_weights_only=True,
                                                      verbose=1)

        # Step 1: Train Policy on D
        policy.fit(X_train_D, y_train_D, epochs=num_epochs,
                   validation_data=(data['X']['test'], data['y']['test']),
                   callbacks=[cp_callback])

        # Step 2: Run Policy in env and collect D_train
        train_acts, train_obs = [], []
        obs = env.reset()
        train_obs.append(obs)
        done = False
        steps = 0
        tot_reward = 0.

        while not done:
            action = expert(obs[None, :].astype('float32'))
            obs, reward, done, _ = env.step(action)
            train_obs.append(obs)
            tot_reward += reward

            steps += 1
            if steps > env.spec.max_episode_steps:
                break

        # Step 3: Label train_obs by expert
        train_acts = [expert(obs[None, :].astype('float32')) for obs in train_obs]

        # Step 4: Add train_obs, train_acts to dataset
        train_obs = np.asarray(train_obs)
        train_acts = np.squeeze(np.asarray(train_acts))

        X_train_D = np.vstack([X_train_D, train_obs])
        y_train_D = np.vstack([y_train_D, train_acts])

        inds = np.random.shuffle(np.arange(X_train_D.shape[0]))
        X_train_D, y_train_D = np.squeeze(X_train_D[inds]), np.squeeze(y_train_D[inds])


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--num_its', type=int, default=10)
    parser.add_argument('--num_train_rolls', type=int, default=20)
    parser.add_argument('--num_epochs', type=int, default=5)
    args = parser.parse_args()

    with open(op.join('expert_data', args.envname + '_' + '{}_eps.pkl'.format(args.num_train_rolls)),
              "rb") as input_file:
        train_data = pickle.load(input_file)

    print("Observations : ", train_data['observations'].shape)
    print("Actions : ", train_data['actions'].shape)

    data = {}
    data['X'], data['y'] = {}, {}
    data['X']['train'], data['X']['test'], data['y']['train'], data['y']['test'] = train_test_split(
        train_data['observations'], np.squeeze(train_data['actions']), test_size=0.2)

    # Store checkpoints as well
    net = models.Net(obs_size=train_data['observations'].shape[1],
                     hidden=256,
                     act_space_size=train_data['actions'].shape[2])

    net.compile(optimizer='adam', loss='mean_squared_error')

    env = gym.make(args.envname)
    max_steps = env.spec.max_episode_steps
    policy_fn = load_policy.load_policy(args.expert_policy_file)

    train_dagger(env, expert=policy_fn, policy=net, num_its=args.num_its,
                 num_epochs=args.num_epochs, data=data,
                 cp_dir='checkpoints/Dagger',
                 cp_name='{}_train_rolls_{}_epochs_'.format(args.num_train_rolls, args.num_epochs))


if __name__ == '__main__':
    main()
