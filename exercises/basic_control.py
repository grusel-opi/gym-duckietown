# !/usr/bin/env python3
"""
Simple exercise to construct a controller that controls the simulated Duckiebot using pose.
"""

import time
import sys
import argparse
import math
import numpy as np
import gym
import functools
from gym_duckietown.envs import DuckietownEnv
from lokalisierung.MCL import MCL

from lokalisierung.Particle import Particle
from lokalisierung.Ducky_map import DuckieMap

parser = argparse.ArgumentParser()
parser.add_argument('--env-name', default=None)
parser.add_argument('--map-name', default='udem1')
parser.add_argument('--no-pause', action='store_true', help="don't pause on failure")
args = parser.parse_args()

if args.env_name is None:
    env = DuckietownEnv(
        map_name=args.map_name,
        domain_rand=False,
        draw_bbox=False
    )
else:
    env = gym.make(args.env_name)

obs = env.reset()
env.render()

total_reward = 0
px, _, py = env.cur_pos
my_map = DuckieMap("../gym_duckietown/maps/udem1.yaml")
particle_number = 1000
mcl = MCL(particle_number, my_map, env)
mcl.spawn_particle_list(env.cur_pos, env.cur_angle)
step_counter = 0
while True:
    lane_pose = env.get_lane_pos2(env.cur_pos, env.cur_angle)
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

    # No need to edit code below.
    obs, reward, done, info = env.step([speed, steering])
    mcl.integrate_movement([speed, steering])
    step_counter += 1
    mcl.integrate_measurement(distance_to_road_center, angle_from_straight_in_rads)
    if step_counter % 10 == 0:
        start = time.time()
        arr_chosenones, possible_location, possible_angle = mcl.resampling()
        end = time.time()
        duration = end- start
        filename = 'data' + str(particle_number) + 'Particles.txt'
        with open(filename, 'a') as f:
            f.write(str(duration) + ' ')
            f.write(str(possible_location[0] - env.cur_pos[0]) + ' ')
            f.write(str(possible_location[2] - env.cur_pos[2]) + ' ')
            f.write(str(possible_angle - env.cur_angle) + ' \n')
        print("posloc and robot position", possible_location, env.cur_pos)
        print('possible angle and robot angle', possible_angle, env.cur_angle)
        mcl.weight_reset()

    total_reward += reward

    print('Steps = %s, Timestep Reward=%.3f, Total Reward=%.3f' % (env.step_count, reward, total_reward))
    print('Distance to road center: ', distance_to_road_center)

    env.render()

    if step_counter == 1000:
        print('*** DONE ***')
        break

    if done:
        if reward < 0:
            print('*** CRASHED ***')
        print('Final Reward = %.3f' % total_reward)
        break
