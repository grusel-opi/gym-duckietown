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
pangle = np.rad2deg(env.cur_angle)
my_map = DuckieMap("../gym_duckietown/maps/udem1.yaml")
aParticle = Particle(px, py, 1, 'p1', my_map, angle=pangle)
aParticle.set_tile()
print('start particle position und tile', aParticle.p_x, aParticle.p_y, aParticle.tile.type)
mcl = MCL(100, my_map)
mcl.spawn_particle_list()
print("length of p_list", len(mcl.p_list))
step_counter = 0
while True:
    lane_pose = env.get_lane_pos2(env.cur_pos, env.cur_angle)
    distance_to_road_center = lane_pose.dist
    angle_from_straight_in_rads = lane_pose.angle_rad

    ###### Start changing the code here.
    # TODO: Decide how to calculate the speed and direction.

    k_p = 10
    k_d = 1

    #aParticle.weight_calculator(1, 1)

    # The speed is a value between [0, 1] (which corresponds to a real speed between 0m/s and 1.2m/s)

    speed = 0.2  # TODO: You should overwrite this value

    # angle of the steering wheel, which corresponds to the angular velocity in rad/s
    steering = k_p * distance_to_road_center + k_d * angle_from_straight_in_rads  # TODO: You should overwrite this value

    ###### No need to edit code below.
    obs, reward, done, info = env.step([speed, steering])
    aParticle.step([speed, steering])
    dist_robot = aParticle.distance_to_wall()
    ang_robot = aParticle.angle_to_wall()
    print(dist_robot, ang_robot, aParticle.p_x, aParticle.p_y, aParticle.tile.type)
    mcl.weight_particles([speed, steering], dist_robot, ang_robot)
    step_counter += 1
    if step_counter % 10 == 0:
        arr_chosenones = mcl.resampling()
        for x in arr_chosenones:
            if x.tile.type in ['floor', 'asphalt', 'grass']:
                print("x tile in choosenones",x)
        sum_py = 0
        sum_px = 0
        for x in arr_chosenones:
            sum_px = sum_px + x.p_x
            sum_py = sum_py + x.p_y
        #sum_px = functools.reduce(lambda a,b : a.p_x + b.p_x, arr_chosenones)
        #sum_py = functools.reduce(lambda a,b : a.p_y + b.p_y, arr_chosenones)
        possible_location = [sum_px / len(arr_chosenones), 0, sum_py / len(arr_chosenones)]
        print('posloc',possible_location,'aParticle',aParticle.p_x, aParticle.p_y)

    # print('particle in duckietown',aParticle.step([speed,0]))

    total_reward += reward

    print('Steps = %s, Timestep Reward=%.3f, Total Reward=%.3f' % (env.step_count, reward, total_reward))
    print('Distance to road center: ', distance_to_road_center)

    env.render()

    if done:
        if reward < 0:
            print('*** CRASHED ***')
        print('Final Reward = %.3f' % total_reward)
        break
