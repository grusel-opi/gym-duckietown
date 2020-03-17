#!/usr/bin/env python

"""
This script allows you to manually control the simulator or Duckiebot
using the keyboard arrows.
"""

import numpy as np
import sys
import argparse
from pyglet import app, clock
from pyglet.window import key
import gym
import gym_duckietown
from gym_duckietown.envs import DuckietownEnv
from gym_duckietown.simulator import Simulator

parser = argparse.ArgumentParser()
parser.add_argument('--env-name', default=None)
parser.add_argument('--map-name', default='udem1')
parser.add_argument('--draw-curve', action='store_true', help='draw the lane following curve')
parser.add_argument('--draw-bbox', action='store_true', help='draw collision detection bounding boxes')
parser.add_argument('--domain-rand', action='store_true', help='enable domain randomization')
parser.add_argument('--frame-skip', default=1, type=int, help='number of frames to skip')
args = parser.parse_args()

RESET_POSE = np.array([[85, 0, 0], [0, 5, 0]])

if args.env_name is None:
    env0 = DuckietownEnv(
        map_name=args.map_name,
        draw_curve=args.draw_curve,
        draw_bbox=args.draw_bbox,
        domain_rand=args.domain_rand,
        frame_skip=args.frame_skip
    )
    env1 = DuckietownEnv(
        map_name=args.map_name,
        draw_curve=args.draw_curve,
        draw_bbox=args.draw_bbox,
        domain_rand=args.domain_rand,
        frame_skip=args.frame_skip
    )
else:
    env0 = gym.make(args.env_name)
    env1 = gym.make(args.env_name)

env0.reset()
env1.reset()
env0.render()
env1.render()

assert isinstance(env0.unwrapped, Simulator)
assert isinstance(env1.unwrapped, Simulator)

@env0.unwrapped.window.event
def on_key_press(symbol, modifiers):
    """
    This handler processes keyboard commands that control the simulation
    """

    if symbol == key.BACKSPACE or symbol == key.SLASH:
        env0.reset()
        env0.render()
        return
    elif symbol == key.ESCAPE:
        env0.close()
        sys.exit(0)
    elif symbol == key.R:
        cam_offset, cam_angle = env0.unwrapped.cam_offset, env0.unwrapped.cam_angle
        cam_angle[0] = RESET_POSE[0][0]
        cam_angle[1] = RESET_POSE[0][1]
        cam_angle[2] = RESET_POSE[0][2]
        cam_offset[0] = RESET_POSE[1][0]
        cam_offset[1] = RESET_POSE[1][1]
        cam_offset[2] = RESET_POSE[1][2]

    # Camera movement
    cam_offset, cam_angle = env0.unwrapped.cam_offset, env0.unwrapped.cam_angle
    if symbol == key.W:
        cam_angle[0] -= 5
    elif symbol == key.S:
        cam_angle[0] += 5
    elif symbol == key.A:
        cam_angle[1] -= 5
    elif symbol == key.D:
        cam_angle[1] += 5
    elif symbol == key.Q:
        cam_angle[2] -= 5
    elif symbol == key.E:
        cam_angle[2] += 5
    elif symbol == key.UP:
        if modifiers:  # Mod+Up for height
            cam_offset[1] += 1
        else:
            cam_offset[0] += 1
    elif symbol == key.DOWN:
        if modifiers:  # Mod+Down for height
            cam_offset[1] -= 1
        else:
            cam_offset[0] -= 1
    elif symbol == key.LEFT:
        cam_offset[2] -= 1
    elif symbol == key.RIGHT:
        cam_offset[2] += 1
    print("cam_angle: ", cam_angle)
    print("cam_offset: ", cam_offset)

    # Take a screenshot
    # UNCOMMENT IF NEEDED - Skimage depencency
    # elif symbol == key.RETURN:
    #     print('saving screenshot')
    #     img = env.render('rgb_array')
    #     try:
    #         from experiments.utils import save_img
    #         save_img('screenshot.png', img)
    #     except BaseException as e:
    #         print(str(e))

def update(dt):
    env0.render('free_cam')
    env1.render('free_cam')


# Main event loop
clock.schedule_interval(update, 1.0 / env0.unwrapped.frame_rate)
clock.schedule_interval(update, 1.0 / env1.unwrapped.frame_rate)
app.run()

env0.close()
env1.close()
