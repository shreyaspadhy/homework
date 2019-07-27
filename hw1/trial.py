# Snippets of code to learn OpenAI Gym Syntax
# Author: Shreyas Padhy
# Date: 5 Jun 2019

# %% Hello World in OpenAI
import tensorflow.keras.layers as layers
import tensorflow as tf
import gym

# env = gym.make('MountainCar-v0')
# env.reset()

# for _ in range(1000):
#     env.render()
#     env.step(env.action_space.sample())  # Taking a random action

# env.close()

# # %% Now with non-random actions
# env = gym.make('Hopper-v3')
# for ep_i in range(20):  # Number of episodes to simulate
#     observation = env.reset()  # Returns an initial observation

#     for t in range(10000):    # Max number of actions to take
#         env.render()
#         print(observation)
#         action = env.action_space.sample()  # Random action
#         observation, reward, done, info = env.step(action)
#         if done:
#             print("Whoops, it's done in {} timesteps".format(t + 1))
#             break

# env.close()

# %% Tensorflow 2.0 Tutorials

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.Sequential([layers.Flatten(input_shape=(x_train.shape[1], x_train.shape[2])),
                             layers.Dense(128, activation='relu'),
                             layers.Dropout(0.2),
                             layers.Dense(10, activation='softmax')
                             ])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)
# %%
