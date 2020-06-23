#!/usr/bin/env python3

"""
Simple exercise to construct a controller that controls the simulated Duckiebot using pose.
"""

import cv2
import sys
import numpy as np
import argparse
import gym
import os
from gym_duckietown.envs import DuckietownEnv
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

RESIZE_IMG_SHAPE = (120, 160, 3)


def preprocess(image):
    height, width, _ = RESIZE_IMG_SHAPE
    image = tf.image.resize(image, (height, width))
    image = tf.image.crop_to_bounding_box(image,
                                          offset_height=height // 2,
                                          offset_width=0,
                                          target_height=height // 2,
                                          target_width=width)
    image = image / 255.
    return np.array([image])


env = DuckietownEnv()
obs = env.reset()

obs = preprocess(obs)

env.render()
total_reward = 0
model = tf.keras.models.load_model('./saved_model/22.06.2020-12:09:34/')

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
