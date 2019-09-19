import os
import sys
import random
import argparse
import signal

import warnings

import numpy as np
import gym
import cv2

import pygame

import matplotlib.pyplot as plt

import skimage as skimage
from skimage import transform, color, exposure
from skimage.transform import rotate
from skimage.viewer import ImageViewer

from collections import deque
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.initializers import normal, identity
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import tensorflow as tf
from tensorflow.keras import backend as K

import gym_donkeycar

import game

EPISODES = 10000

img_rows = img_cols = 70

img_channels = 4

tf.logging.set_verbosity(tf.logging.ERROR)

global image
image = None

class agent:

    def __init__(self, state_size, action_space, train = True):
        self.t = 0
        self.max_Q = 0
        self.train = train
        
        self.action_space = action_space
        self.action_size = action_space

        self.discount_factor = 0.99
        self.learning_rate = 0.0001

        if self.train:
            self.epsilon = 1.0
            self.initial_epsilon = 1.0
        else:
            self.epsilon = 1e-6
            self.initial_epsilon = 1e-6
        self.epsilon_min = 0.02
        self.batch_size = 64
        self.train_start = 100
        self.explore = 10000

        self.memory = deque(maxlen=10000)

        self.model = self.build_model()

        self.target_model = self.build_model()

        self.update_target_model()

    def build_model(self):
        model = Sequential()
        model.add(Conv2D(24, (5, 5), strides=(2, 2), padding="same",input_shape=(img_rows,img_cols,img_channels)))  #80*80*4
        model.add(Activation('relu'))
        model.add(Conv2D(32, (5, 5), strides=(2, 2), padding="same"))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (5, 5), strides=(2, 2), padding="same"))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3), strides=(2, 2), padding="same"))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3), strides=(1, 1), padding="same"))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))

        # 15 categorical bins for Steering angles
        model.add(Dense(25, activation="linear")) 

        adam = Adam(lr=self.learning_rate)
        model.compile(loss='mse',optimizer=adam)
        
        return model

    def process_image(self, obs):

        global image

        result = cv2.resize(obs, (img_rows, img_cols))

        # if image == None:
        #     image = plt.imshow(result)
        # else:
        #     image.set_data(result)
        # plt.pause(0.1)
        # plt.draw()
        # # plt.imshow(result)
        # # plt.show()
        return result
        #return cv2.resize(obs, (img_rows, img_cols))

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())


    def get_action(self, s_t):
        if np.random.rand() <= self.epsilon:
            return random.sample(range(len(self.action_space)), 1)[0]
            #return np.random.rand() - 0.5
        else:
            q_value = self.model.predict(s_t)
            #print(np.argmax(q_value))
            return np.argmax(q_value)


    def replay_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon -= (self.initial_epsilon - self.epsilon_min) / self.explore

    def train_replay(self):

        if len(self.memory) < self.train_start:
            return

        batch_size = min(self.batch_size, len(self.memory))

        minibatch = random.sample(self.memory, batch_size)

        state_t, action_t, reward_t, state_t1, terminal = zip(*minibatch)

        state_t = np.concatenate(state_t)
        state_t1 = np.concatenate(state_t1)

        targets = self.model.predict(state_t)

        self.max_Q = np.max(targets[0])

        target_val = self.model.predict(state_t1)

        target_val_ = self.target_model.predict(state_t1)

        for i in range(batch_size):
            if terminal[i]:
                targets[i][action_t[i]] = reward_t[i]
            else:
                a = np.argmax(target_val[i])
                targets[i][action_t[i]] = reward_t[i] + self.discount_factor * (target_val_[i][a])
        
        self.model.train_on_batch(state_t, targets)

    def load_model(self, name):
        self.model.load_weights(name)

    def save_model(self, name):
        warnings.filterwarnings("ignore")
        self.model.save_weights(name)
        warnings.filterwarnings("default")

def linear_bin(a):
    
    a = (a * 10) + 1
    b = round(a * 7)
    arr = np.zeros(15)
    arr[int(b)] = 1

    return arr

def linear_unbin(arr):

    b = np.argmax(arr) 
    interval = np.arange(15) / 70 - 0.10
    return interval[b]

def run_ddqn():

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    sess = tf.Session(config = config)

    K.set_session(sess)

    state_size = (img_rows, img_cols, img_channels)

    s = np.linspace(-0.1, 0.1, 5)
    t = np.linspace(-1, 3, 5)

    mesh = np.meshgrid(s, t)

    action_space = np.array(list(zip(mesh[0].flatten().tolist(), mesh[1].flatten().tolist())))

    env = game.game()

    a = agent(state_size, action_space, train = True)

    episodes = []

    global image

    image = None

    for e in range(EPISODES):

        #pygame.init()
        #display = pygame.display.set_mode((img_rows, img_cols))

        print("Epsode: ", e)

        done = False

        obs = env.reset()

        episode_len = 0

        x_t = a.process_image(obs)

        s_t = np.stack((x_t, x_t, x_t, x_t), axis = 2)

        s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2])

        # im = plt.imshow(x_t)
        # plt.show(block = False)

        episode_memory = []

        while not done:

            idx = a.get_action(s_t)

            steering, throttle = action_space[idx]

            new_obs, reward, done = env.step(steering, throttle, x_t.T)

            #surf = pygame.surfarray.make_surface(new_obs)
            #display.blit(surf, (0,0))
            #pygame.display.update()

            x_t1 = a.process_image(new_obs)
            x_t = x_t1
            # im.set_data(x_t1)
            # plt.pause(0.001)
            # plt.draw()

            x_t1 = x_t1.reshape(1, x_t1.shape[0], x_t1.shape[1], 1)

            s_t1 = np.append(x_t1, s_t[:, :, :, :3], axis=3)

            #episode_memory.insert(0, (s_t, idx, reward, s_t1, done))
            
            a.replay_memory(s_t, idx, reward, s_t1, done)

            if a.train:
                a.train_replay()
            
            a.update_epsilon()
            

            s_t = s_t1
            a.t += 1

            episode_len += 1

            if a.t % 10 == 0:
                print("EPISODE",  e, "TIMESTEP", a.t,"/ ACTION", round(steering, 2), " ", round(throttle, 2), "/ REWARD", reward, "/ EPISODE LENGTH", episode_len, "/ Q_MAX " , a.max_Q)

            if done:
                a.update_target_model()

                episodes.append(e)

                if a.train:
                    a.save_model("Malli4")

                print("episode:", e, "  memory length:", len(a.memory),
                        "  epsilon:", a.epsilon, " episode length:", episode_len)

        # for i, m in enumerate(episode_memory):
            
        #     discount = max(0, (1 - i / 4))
        #     a.replay_memory(m[0], m[1], m[2] - discount, m[3], m[4])

        

if __name__ == "__main__":
    run_ddqn()


