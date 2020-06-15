#!/usr/bin/env python3

"""
Simple exercise to construct a controller that controls the simulated Duckiebot using pose.
"""

import cv2
import numpy as np
import argparse
import gym
import os
from gym_duckietown.envs import DuckietownEnv
import tensorflow as tf
from data_loader import TARGET_IMG_HEIGHT, TARGET_IMG_WIDTH

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def preprocess(img):
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.per_image_standardization(img)
    img = tf.image.resize(img, [TARGET_IMG_HEIGHT, TARGET_IMG_WIDTH])
    return np.array([img])


env = DuckietownEnv()
obs = env.reset()

obs = preprocess(obs)

env.render()
total_reward = 0
model = tf.keras.models.load_model('./saved_model/08.06.2020-18:54:20/')
k_p = 10
k_d = 1
speed = 0.2

while True:

    lane_pose = env.get_lane_pos2(env.cur_pos, env.cur_angle)
    distance_to_road_center = lane_pose.dist
    angle_from_straight_in_rads = lane_pose.angle_rad

    d, a = model.predict(obs)

    dist_err = distance_to_road_center - d
    angle_err = angle_from_straight_in_rads - a

    print()
    print(d, a)
    print(distance_to_road_center, angle_from_straight_in_rads)
    print("error: {}, {}".format(dist_err, angle_err))

    steering = k_p * distance_to_road_center + k_d * angle_from_straight_in_rads

    obs, reward, done, info = env.step(np.array([speed, steering]))
    total_reward += reward
    obs = preprocess(obs)

    env.render()

    if done:
        if reward < 0:
            print('*** CRASHED ***')
        print('Final Reward = %.3f' % total_reward)
        break
