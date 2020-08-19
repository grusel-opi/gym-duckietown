#!/usr/bin/env python3

import cv2
import numpy as np
import os
import math
from gym_duckietown.envs import DuckietownEnv
from data_generator import OwnLanePosition
from gym_duckietown.simulator import NotInLane, get_dir_vec
import tensorflow as tf
from pyglet.window import key
import sys
from pid_controller import PID

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

RESIZE_IMG_SHAPE = (120, 160, 3)
DEBUG = False

def get_lane_pos(env):
    pos = env.cur_pos
    angle = env.cur_angle
    point, tangent = env.closest_curve_point(pos, angle)
    if point is None:
        msg = 'Point not in lane: %s' % pos
        raise NotInLane(msg)

    assert point is not None

    dirVec = get_dir_vec(angle)
    dotDir = np.dot(dirVec, tangent)
    dotDir = max(-1, min(1, dotDir))

    posVec = pos - point
    upVec = np.array([0, 1, 0])
    rightVec = np.cross(tangent, upVec)
    signedDist = np.dot(posVec, rightVec)
    dist_to_road_edge = 0.25 * env.road_tile_size - signedDist
    angle_rad = math.acos(dotDir)

    if np.dot(dirVec, rightVec) < 0:
        angle_rad *= -1

    angle_deg = np.rad2deg(angle_rad)

    return OwnLanePosition(dist=signedDist,
                           dist_to_edge=dist_to_road_edge,
                           dot_dir=dotDir,
                           angle_deg=angle_deg,
                           angle_rad=angle_rad)


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
# model = tf.keras.models.load_model('./saved_model/30.06.2020-17:44:24')

k_u = 0.15
p_u = 125
k_p = 0.6 * k_u
k_i = 2 * k_p / p_u
k_d = k_p * p_u / 8

pid = PID(k_p, k_i, k_d, 25)
speed = 0.3

key_handler = key.KeyStateHandler()
env.unwrapped.window.push_handlers(key_handler)

@env.unwrapped.window.event
def on_key_press(symbol, modifiers):
    global DEBUG

    if symbol == key.BACKSPACE:
        print("RESET")
        env.reset()
        env.render()
    elif symbol == key.ESCAPE:
        env.close()
        sys.exit(0)
    elif symbol == key.D:
        DEBUG = not DEBUG


while True:
    dist_to_road_edge = get_lane_pos(env).dist_to_edge * 100
    # pred_dist = model.predict(obs)[0][0]

    correction = pid.update(dist_to_road_edge)
    # correction = pid.update(pred_dist)

    if DEBUG:
        print()
        # print("pred_dist: ", pred_dist)
        print("real_dist: ", dist_to_road_edge)
        # print("dist_err: ", abs(distance_to_edge - pred_dist))
        print("correction: ", correction)

    obs, reward, done, info = env.step(np.array([speed, correction]))
    total_reward += reward
    # obs = preprocess(obs)

    env.render()
