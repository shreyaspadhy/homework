#!/usr/bin/env python

"""
Code for Behaviour Cloning
Example usage:
    python behaviour_cloning.py ./expert_data/Humanoid-v2.pkl Humanoid-v2.pkl --num_train_rolls 20
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


def train_BC(env, policy, num_epochs=5, data={},
             cp_dir='checkpoints/BC', cp_name=''):
    # Create checkpoint callback

    checkpoint_path = "{}/{}/{}/cp.ckpt".format(cp_dir, env.unwrapped.spec.id, cp_name)
    checkpoint_dir = os.path.dirname(checkpoint_path)
    os.makedirs(checkpoint_dir, exist_ok=True)

    cp_callback = keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                  save_weights_only=True,
                                                  verbose=1)

    # Step 1: Train Policy on D
    policy.fit(data['X']['train'], data['y']['train'], epochs=num_epochs,
               validation_data=(data['X']['test'], data['y']['test']),
               callbacks=[cp_callback])


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
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

    train_BC(env, policy=net,
             num_epochs=args.num_epochs, data=data,
             cp_dir='checkpoints/BC',
             cp_name='{}_train_rolls_{}_epochs'.format(args.num_train_rolls, args.num_epochs))


if __name__ == '__main__':
    main()
