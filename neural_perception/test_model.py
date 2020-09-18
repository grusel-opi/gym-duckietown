#!/usr/bin/env python3

import argparse
import math
import numpy as np
import cv2
import os
import gym
from gym_duckietown.simulator import NotInLane, get_dir_vec
from data_generator import OwnLanePosition
from gym_duckietown.envs import DuckietownEnv
import pyglet
from pyglet.window import key
from pid_controller import PID

RESIZE_IMG_SHAPE = (120, 160, 3)
DEBUG = False
MANUAL_CONTROL = False

parser = argparse.ArgumentParser()
parser.add_argument('--env-name', default=None)
parser.add_argument('--map-name', default='udem1')
parser.add_argument('--no-pause', action='store_true', help="don't pause on failure")
args = parser.parse_args()

if args.env_name is None:
    env = DuckietownEnv(
        map_name=args.map_name,
        domain_rand=False,
        draw_bbox=False,
    )
else:
    env = gym.make(args.env_name)


def preprocess(image):
    height, width, _ = RESIZE_IMG_SHAPE
    image = cv2.resize(image, (width, height))
    image = image[int(height / 3):, 0:width]
    image = image / 255.
    return np.array([image])


def get_lane_pos(enviroment):
    pos = enviroment.cur_pos
    angle = enviroment.cur_angle
    point, tangent = enviroment.closest_curve_point(pos, angle)
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
    dist_to_road_edge = 0.25 * enviroment.road_tile_size - signedDist
    angle_rad = math.acos(dotDir)

    if np.dot(dirVec, rightVec) < 0:
        angle_rad *= -1

    angle_deg = np.rad2deg(angle_rad)

    return OwnLanePosition(dist=signedDist,
                           dist_to_edge=dist_to_road_edge,
                           dot_dir=dotDir,
                           angle_deg=angle_deg,
                           angle_rad=angle_rad)


obs = env.reset()
obs = preprocess(obs)
env.render()
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

@env.unwrapped.window.event
def on_key_press(symbol, modifiers):
    global DEBUG
    global MANUAL_CONTROL

    if symbol == key.BACKSPACE:
        print("RESET")
        env.reset()
        env.render()
    elif symbol == key.ESCAPE:
        env.close()
        exit(0)
    elif symbol == key.D:
        DEBUG = not DEBUG
    elif symbol == key.M:
        if not MANUAL_CONTROL:
            print("Manual control activated")
        else:
            print("Manual control deactivated")
        MANUAL_CONTROL = not MANUAL_CONTROL


key_handler = key.KeyStateHandler()
env.unwrapped.window.push_handlers(key_handler)


# model = tf.keras.models.load_model('./saved_model/30.06.2020-17:44:24')
k_u = 0.05
p_u = 125
k_p = 0.6 * k_u
k_i = 2 * k_p / p_u
k_d = k_p * p_u / 8

pid = PID(k_p, k_i, k_d, 0.25)
speed = 0.2
correction = 0
action = np.array([0, correction])


def update(dt):
    global correction
    global action
    global speed

    lane_pos = get_lane_pos(env)
    dist_to_road_edge = lane_pos.dist_to_edge
    # pred_dist = model.predict(obs)[0][0]

    if not MANUAL_CONTROL:
        correction = pid.update(dist_to_road_edge)
        action = np.array([speed, correction])
        # correction = pid.update(pred_dist)
    else:
        action = np.array([0.0, 0.0])

        if key_handler[key.UP]:
            print("YOO")
            action = np.array([0.44, 0.0])
        if key_handler[key.DOWN]:
            action = np.array([-0.44, 0])
        if key_handler[key.LEFT]:
            action = np.array([0.35, +1])
        if key_handler[key.RIGHT]:
            action = np.array([0.35, -1])

    if DEBUG:
        print()
        # print("pred_dist: ", pred_dist)
        print("dist_to_road_edge: ", dist_to_road_edge)
        print("correction: ", correction)
        print("signed_dist: ", lane_pos.dist)
        print("dot_dir: ", lane_pos.dot_dir)
        print("angle_deg: ", lane_pos.angle_deg)
        # print("dist_err: ", abs(distance_to_edge - pred_dist))

    obs, _, _, _ = env.step(action)
    # obs = preprocess(obs)

    env.render()

pyglet.clock.schedule_interval(update, 1.0 / env.unwrapped.frame_rate)
pyglet.app.run()
env.close()

