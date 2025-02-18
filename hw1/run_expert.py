#!/usr/bin/env python

"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python run_expert.py experts/Humanoid-v1.pkl Humanoid-v1 --render \
            --num_rollouts 20

Author of this script and included expert policies: Jonathan Ho (hoj@openai.com)

Changes made to make compatible with TF 2.0 and gym 12.x: Shreyas Padhy (shreyaspadhy@gmail.com)
"""

import os
import pickle
import tensorflow as tf
import numpy as np
import gym
import load_policy


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    args = parser.parse_args()

    print('loading and building expert policy')
    policy_fn = load_policy.load_policy(args.expert_policy_file)
    print('loaded and built')

    env = gym.make(args.envname)
    max_steps = args.max_timesteps or env.spec.max_episode_steps

    returns = []
    observations = []
    actions = []

    for i in range(args.num_rollouts):
        print('iter', i)
        obs = env.reset()
        done = False
        tot_reward = 0.
        steps = 0
        while not done:

            action = policy_fn(obs[None, :].astype('float32'))
            observations.append(obs)
            actions.append(action)
            obs, reward, done, _ = env.step(action)
            tot_reward += reward
            steps += 1
            if args.render:
                env.render()
            if steps % 100 == 0:
                print("%i/%i" % (steps, max_steps))
            if steps >= max_steps:
                break
        returns.append(tot_reward)

    print('returns', returns)
    print('mean return', np.mean(returns))
    print('std of return', np.std(returns))

    expert_data = {'observations': np.array(observations),
                   'actions': np.array(actions)}

    os.makedirs('expert_data', exist_ok=True)
    with open(os.path.join('expert_data', args.envname + '_{}_eps.pkl'.format(args.num_rollouts)), 'wb') as f:
        pickle.dump(expert_data, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
