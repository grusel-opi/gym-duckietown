#!/usr/bin/env python3

"""
Simple exercise to construct a controller that controls the simulated Duckiebot using pose.
"""

import numpy as np
import os
from gym_duckietown.envs import DuckietownEnv
import tensorflow as tf

from neural_perception.util import preprocess, get_lane_pos

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

env = DuckietownEnv(domain_rand=False,
                    draw_bbox=False)

obs = env.reset()

obs = preprocess(obs)

env.render()

model = tf.keras.models.load_model('../../lean-test/saved_model/18.09.2020-13:24:04/')

k_p = 1
k_d = 1
speed = 0.2

steps = env.max_steps = 10_000

for i in range(steps):

    lane_pose = get_lane_pos(env)

    distance_to_road_edge = lane_pose.dist_to_edge * 100
    distance_to_road_center = lane_pose.dist
    angle_from_straight_in_rad = lane_pose.angle_rad
    angle_from_straight_in_deg = lane_pose.angle_deg

    y_hat = model(obs)

    d_hat = y_hat[0][0].numpy()
    a_hat = y_hat[0][1].numpy()

    d_hat_to_center = (d_hat - 25.) / 100.
    a_hat_in_rad = (a_hat * 2 * np.pi) / 360.

    dist_err = abs(distance_to_road_edge - d_hat)
    angle_err = abs(angle_from_straight_in_deg - a_hat)

    print("error: {}, {}".format(dist_err, angle_err))

    steering = k_p * d_hat_to_center + k_d * a_hat_in_rad
    command = np.array([speed, steering])
    obs, _, done, _ = env.step(command)

    obs = preprocess(obs)
    env.render()

    if done:
        env.reset()
