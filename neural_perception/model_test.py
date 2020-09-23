#!/usr/bin/env python3

import argparse
import numpy as np
import os
import gym
from gym_duckietown.envs import DuckietownEnv
import pyglet
from pyglet.window import key
from neural_perception.pid_controller import PID
from neural_perception.util import preprocess, get_lane_pos

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

