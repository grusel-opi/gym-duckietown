#!/usr/bin/env python3

"""
Simple exercise to construct a controller that controls the simulated Duckiebot using pose.
"""

import numpy as np
import os
from gym_duckietown.envs import DuckietownEnv
import tensorflow as tf

from neural_perception.util.util import preprocess, get_lane_pos

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

env = DuckietownEnv(domain_rand=False,
                    draw_bbox=False)

obs = env.reset()

obs = preprocess(obs)

env.render()

model = tf.keras.models.load_model('/home/gandalf/ws/team/gym-duckietown/neural_perception/model/saved_model/05.10.2020-16:00:31')

steps = env.max_steps = 10_000

for i in range(steps):

    action = model(obs)[0].numpy()

    print("\raction: {}, {}".format(action[0], action[1]), end='\r')

    action[0] = 0.1

    obs, _, done, _ = env.step(action)

    obs = preprocess(obs)
    env.render()

    if done:
        env.reset()
