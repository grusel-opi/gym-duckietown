#!/usr/bin/env python
# manual

"""
This script allows you to manually control the simulator or Duckiebot
using the keyboard arrows.
"""

import sys
import pyglet
import numpy as np
import tensorflow as tf

from pyglet.window import key
from gym_duckietown.envs import DuckietownEnv
from neural_perception.util.util import preprocess, get_lane_pos

env = DuckietownEnv(domain_rand=False,
                    draw_bbox=False)

model = tf.keras.models.load_model('../../duckiepose/saved_model/18.09.2020-13:24:04/', compile=False)

env.reset()
env.render()

steps = env.max_steps = 10_000


@env.unwrapped.window.event
def on_key_press(symbol, modifiers):
    if symbol == key.BACKSPACE or symbol == key.SLASH:
        print('RESET')
        env.reset()
        env.render()
    elif symbol == key.PAGEUP:
        env.unwrapped.cam_angle[0] = 0
    elif symbol == key.ESCAPE:
        env.close()
        sys.exit(0)


key_handler = key.KeyStateHandler()
env.unwrapped.window.push_handlers(key_handler)


def update(dt):
    """
    This function is called at every frame to handle
    movement/stepping and redrawing
    """

    action = np.array([0.0, 0.0])

    if key_handler[key.UP]:
        action = np.array([0.44, 0.0])
    if key_handler[key.DOWN]:
        action = np.array([-0.44, 0])
    if key_handler[key.LEFT]:
        action = np.array([0.35, +1])
    if key_handler[key.RIGHT]:
        action = np.array([0.35, -1])
    if key_handler[key.SPACE]:
        action = np.array([0, 0])

    # Speed boost
    if key_handler[key.LSHIFT]:
        action *= 1.5

    obs, reward, done, info = env.step(action)

    lane_pose = get_lane_pos(env)

    distance_to_road_edge = lane_pose.dist_to_edge * 100
    distance_to_road_center = lane_pose.dist
    angle_from_straight_in_rad = lane_pose.angle_rad
    angle_from_straight_in_deg = lane_pose.angle_deg

    y_hat = model(preprocess(obs))
    d = y_hat[0][0].numpy()
    a = y_hat[0][1].numpy()

    dist_err = abs(distance_to_road_edge - d)
    angle_err = abs(angle_from_straight_in_deg - a)

    print(
        f"\ractu: {round(distance_to_road_edge, 1)}, {round(angle_from_straight_in_deg, 1)}, pred: {round(d, 1)}, {round(a, 1)}, error: {round(dist_err, 1)}, {round(angle_err, 1)}",
        end='\r')

    if done:
        print('done!')
        env.reset()
        env.render()

    env.render()


pyglet.clock.schedule_interval(update, 1.0 / env.unwrapped.frame_rate)

pyglet.app.run()

env.close()
