#!/usr/bin/env python3

"""
Simple exercise to construct a controller that controls the simulated Duckiebot using pose.
"""

import numpy as np
import os
from gym_duckietown.envs import DuckietownEnv
import tensorflow as tf

from neural_perception.util import preprocess

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

if __name__ == '__main__':
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

        y_hat = model(obs)

        d_hat = y_hat[0][0]
        a_hat = y_hat[0][1]

        distance_to_road_center = (d_hat - 25.) / 100.
        angle_from_straight_in_rad = (a_hat*2*np.pi) / 360.

        steering = k_p * distance_to_road_center + k_d * angle_from_straight_in_rad
        command = np.array([speed, steering])
        obs, _, done, _ = env.step(command)

        obs = preprocess(obs)
        env.render()

        if done:
            env.reset()
