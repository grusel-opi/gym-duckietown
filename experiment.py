#!/usr/bin/env python3

"""
Simple exercise to construct a controller that controls the simulated Duckiebot using pose. 
"""

import time
import sys
import argparse
import math
import numpy as np
import gym
from pyglet import app, clock
from pyglet.window import key
from gym_duckietown.envs import DuckietownEnv

parser = argparse.ArgumentParser()
parser.add_argument('--env-name', default=None)
parser.add_argument('--map-name', default='udem1')
parser.add_argument('--no-pause', action='store_true', help="don't pause on failure")
args = parser.parse_args()

if args.env_name is None:
    env0 = DuckietownEnv(
        map_name=args.map_name,
        domain_rand=False,
        draw_bbox=False
    )
    env1 = DuckietownEnv(
        map_name=args.map_name,
        domain_rand=False,
        draw_bbox=False
    )
else:
    env0 = gym.make(args.env_name)
    env1 = gym.make(args.env_name)

obs = env0.reset()
obs = env1.reset()

env0.render()
env1.render()

RESET_POSE = np.array([[85, 0, 0], [0, 5, 0]])

total_reward = 0


@env1.unwrapped.window.event
def on_key_press(symbol, modifiers):
    """
    This handler processes keyboard commands that control the simulation
    """

    if symbol == key.BACKSPACE or symbol == key.SLASH:
        env1.reset()
        env1.render()
        return
    elif symbol == key.ESCAPE:
        env1.close()
        sys.exit(0)
    elif symbol == key.R:
        cam_offset, cam_angle = env1.unwrapped.cam_offset, env1.unwrapped.cam_angle
        cam_angle[0] = RESET_POSE[0][0]
        cam_angle[1] = RESET_POSE[0][1]
        cam_angle[2] = RESET_POSE[0][2]
        cam_offset[0] = RESET_POSE[1][0]
        cam_offset[1] = RESET_POSE[1][1]
        cam_offset[2] = RESET_POSE[1][2]

    # Camera movement
    cam_offset, cam_angle = env1.unwrapped.cam_offset, env1.unwrapped.cam_angle
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


def update(dt):
    env1.render('free_cam')


clock.schedule_interval(update, 1.0 / env1.unwrapped.frame_rate)
app.run()

while True:

    lane_pose = env0.get_lane_pos2(env0.cur_pos, env0.cur_angle)
    distance_to_road_center = lane_pose.dist
    angle_from_straight_in_rads = lane_pose.angle_rad

    ###### Start changing the code here.
    # TODO: Decide how to calculate the speed and direction.

    k_p = 10
    k_d = 1

    # The speed is a value between [0, 1] (which corresponds to a real speed between 0m/s and 1.2m/s)

    speed = 0.2  # TODO: You should overwrite this value

    # angle of the steering wheel, which corresponds to the angular velocity in rad/s
    steering = k_p * distance_to_road_center + k_d * angle_from_straight_in_rads  # TODO: You should overwrite this value

    ###### No need to edit code below.

    obs, reward, done, info = env0.step([speed, steering])
    total_reward += reward

    print('Steps = %s, Timestep Reward=%.3f, Total Reward=%.3f' % (env0.step_count, reward, total_reward))

    env0.render()

    if done:
        if reward < 0:
            print('*** CRASHED ***')
        print('Final Reward = %.3f' % total_reward)
        break
