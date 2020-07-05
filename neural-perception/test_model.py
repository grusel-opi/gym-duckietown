#!/usr/bin/env python3

"""
Simple exercise to construct a controller that controls the simulated Duckiebot using pose.
"""

import cv2
import numpy as np
import os
from gym_duckietown.envs import DuckietownEnv
import tensorflow as tf
from pid_controller import PID

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

RESIZE_IMG_SHAPE = (120, 160, 3)

def preprocess(image):
    height, width, _ = RESIZE_IMG_SHAPE
    image = cv2.resize(image, (width, height))
    image = image[0:int(height / 2), 0:width]
    image = image / 255.
    return np.array([image])


env = DuckietownEnv()
obs = env.reset()

obs = preprocess(obs)

env.render()
total_reward = 0
model = tf.keras.models.load_model('./saved_model/30.06.2020-15:46:59')

pid = PID(1.0, 1.0, 1.0, 20)
speed = 0.2

while True:

    distance_to_edge = env.get_own_lane_pos(env.cur_pos, env.cur_angle).dist * 100
    # pred_dist = model.predict(obs)[0][0]

    # correction = pid.update(pred_dist)
    correction = pid.update(distance_to_edge)


    # print()
    # print("pred_dist: ", pred_dist)
    # print("real_dist: ", distance_to_edge)
    # print("dist_err: ", abs(distance_to_edge - pred_dist))
    # print("correction: ",s correction)

    obs, reward, done, info = env.step(np.array([speed, correction]))
    total_reward += reward
    obs = preprocess(obs)

    env.render()

    if done:
        if reward < 0:
            print('*** CRASHED ***')
        print('Final Reward = %.3f' % total_reward)
        break
