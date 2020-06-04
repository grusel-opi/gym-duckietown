#!/usr/bin/env python3

"""
Simple exercise to construct a controller that controls the simulated Duckiebot using pose.
"""

import cv2
import numpy as np
import argparse
import gym
from gym_duckietown.envs import DuckietownEnv
import tensorflow as tf
from data_loader import TARGET_IMG_HEIGHT, TARGET_IMG_WIDTH


def preprocess(data):
    return np.array([cv2.resize(data, (TARGET_IMG_WIDTH, TARGET_IMG_HEIGHT))])


env = DuckietownEnv()
obs = env.reset()

print(obs.shape)

obs = preprocess(obs)

print(obs.shape)

env.render()
total_reward = 0
model = tf.keras.models.load_model('./saved_model/04.06.2020-13:31:41/')
k_p = 10
k_d = 1
speed = 0.2

while True:

    prediction = model.predict(obs)
    distance_to_road_center = prediction[0]
    angle_from_straight_in_rads = prediction[1]

    print(env.cur_pos, env.cur_angle)

    steering = k_p * distance_to_road_center + k_d * angle_from_straight_in_rads

    obs, reward, done, info = env.step(np.array([speed, steering]))
    total_reward += reward
    obs = preprocess(obs)

    print('Steps = %s, Timestep Reward=%.3f, Total Reward=%.3f' % (env.step_count, reward, total_reward))

    env.render()

    if done:
        if reward < 0:
            print('*** CRASHED ***')
        print('Final Reward = %.3f' % total_reward)
        break
