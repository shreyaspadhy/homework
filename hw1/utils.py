#!/usr/bin/env python

"""
Utility functions for hw1

Author of this script: Shreyas Padhy (shreyaspadhy@gmail.com)
"""

import os
import os.path as op
import numpy as np
import gym
import load_policy


def run_policy(env, policy_fn, policy='Expert'):
    # Behaviour Cloning
    obs = env.reset()
    done = False
    steps = 0
    tot_reward = 0.

    while not done:
        action = 0.0
        if policy == 'Expert':
            action = policy_fn(obs[None, :].astype('float32'))
        elif policy == 'BC' or policy == 'Dagger':
            action = policy_fn.call(obs[None, :])
        obs, reward, done, _ = env.step(action)
        tot_reward += reward

        steps += 1
        if steps > env.spec.max_episode_steps:
            break

    return tot_reward
