import os
import os.path as op
import pickle
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import numpy as np
import gym
import load_policy


class Net(keras.Model):

    def __init__(self, obs_size=100, hidden=100, act_space_size=10):
        super(Net, self).__init__()
        self.in_layer = keras.models.Sequential([layers.Dense(obs_size, activation='relu'),
                                                 layers.Dropout(0.2)])
        self.hidden_layers = keras.models.Sequential([layers.Dense(hidden, activation='relu'),
                                                      layers.Dropout(0.2),
                                                      layers.Dense(hidden, activation='relu'),
                                                      layers.Dropout(0.2)])
        self.out_layer = layers.Dense(act_space_size, activation='tanh')

    def call(self, obs):
        x = self.in_layer(obs)
        x = self.hidden_layers(x)
        act = self.out_layer(x)
        return act
