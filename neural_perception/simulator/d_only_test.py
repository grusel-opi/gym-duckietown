#!/usr/bin/env python3

import numpy as np
import os
from gym_duckietown.envs import DuckietownEnv
import pyglet
from pyglet.window import key
from neural_perception.util.pid_controller import PID, calculate_out_lim
from neural_perception.util.util import preprocess, get_lane_pos
import tensorflow as tf

env = DuckietownEnv(domain_rand=False,
                    draw_bbox=False)

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
            pid.reset()
        MANUAL_CONTROL = not MANUAL_CONTROL


key_handler = key.KeyStateHandler()
env.unwrapped.window.push_handlers(key_handler)

DEBUG = False
MANUAL_CONTROL = False
model = tf.keras.models.load_model(
    '/home/gandalf/ws/team/gym-duckietown/neural_perception/model/saved_model/17.09.2020-13:40:11')
speed = 0.3
target = 25

k_p_cheat = 0.075
k_i_cheat = 0.0075
k_d_cheat = 0.1875

k_p_modell = 0.075
k_i_modell = 0.00075
k_d_modell = 0.000075

pid = PID(k_p_modell, k_i_modell, k_d_modell, target)
out_lim = calculate_out_lim(env, speed)
pid.set_out_lim(out_lim)
correction = 0


def update(dt):
    global obs
    global correction

    lane_pos = get_lane_pos(env)
    dist_to_road_edge = lane_pos.dist_to_edge
    pred_dist = model.predict(obs)[0][0]

    if not MANUAL_CONTROL:
        # correction = pid.update(dist_to_road_edge * 100)
        correction = pid.update(pred_dist)
        action = np.array([speed, correction])
    else:
        action = np.array([0.0, 0.0])

        if key_handler[key.UP]:
            action = np.array([0.44, 0.0])
        if key_handler[key.DOWN]:
            action = np.array([-0.44, 0])
        if key_handler[key.LEFT]:
            action = np.array([0.35, +1])
        if key_handler[key.RIGHT]:
            action = np.array([0.35, -1])

    if DEBUG:
        print()
        print("pred_dist: ", pred_dist)
        print("dist_to_road_edge: ", dist_to_road_edge)
        print("correction: ", correction)
        # print("out_lim: ", out_lim)
        # print("signed_dist: ", lane_pos.dist)
        # print("dot_dir: ", lane_pos.dot_dir)
        # print("angle_deg: ", lane_pos.angle_deg)
        print("dist_err: ", abs(dist_to_road_edge * 100 - pred_dist))

    obs, _, _, _ = env.step(action)
    obs = preprocess(obs)

    env.render()


pyglet.clock.schedule_interval(update, 1.0 / env.unwrapped.frame_rate)
pyglet.app.run()
env.close()
