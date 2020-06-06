import time
import sys
import argparse
import math
import numpy as np
import gym
from gym_duckietown.envs import DuckietownEnv
import threading

#include <X11/Xlib.h>



parser = argparse.ArgumentParser()
parser.add_argument('--env-name', default=None)
parser.add_argument('--map-name', default='udem1')
parser.add_argument('--no-pause', action='store_true', help="don't pause on failure")
args = parser.parse_args()


if args.env_name is None:
    env = DuckietownEnv(
        map_name = args.map_name,
        domain_rand = False,
        draw_bbox = False
    )
else:
    env = gym.make(args.env_name)

class Particle:
    @staticmethod
    def makeParticle():

        def __init__(self, speed, steering):
            #self.x_global = x_global
            #self.y_global = y_global
            #self.lane_angle_rad = lane_angle_rad
            self.speed = speed
            self.steering = steering

        env_new = DuckietownEnv(DuckietownEnv(
            map_name = args.map_name,
            domain_rand = False,
            draw_bbox = False
        ))
        obs = env.reset()
        env_new.reset()
        env.render()
        n = 0
        total_reward = 0
        while True:

            lane_pose_new = env_new.get_lane_pos2(env_new.cur_pos, env_new.cur_angle)
            print('env_new lane_position: ', lane_pose_new)
            distance_to_road_center_new = lane_pose_new.dist
            print('distance_to_road_center_new: ', distance_to_road_center_new)

            lane_pose = env.get_lane_pos2(env.cur_pos, env.cur_angle)
            distance_to_road_center = lane_pose.dist
            angle_from_straight_in_rads = lane_pose.angle_rad

            ###### Start changing the code here.
            # TODO: Decide how to calculate the speed and direction.

            k_p = 10
            k_d = 1

            env_new.reset()

            # The speed is a value between [0, 1] (which corresponds to a real speed between 0m/s and 1.2m/s)

            speed = 0.2  # TODO: You should overwrite this value

            # angle of the steering wheel, which corresponds to the angular velocity in rad/s
            steering = k_p * distance_to_road_center + k_d * angle_from_straight_in_rads  # TODO: You should overwrite this value

            ###### No need to edit code below.

            obs, reward, done, info = env.step([speed, steering])

            #obs1, reward1, done1, info1 = env_new.step([speed, steering])

            total_reward += reward

            print('Steps = %s, Timestep Reward=%.3f, Total Reward=%.3f' % (env.step_count, reward, total_reward))
            print('Distance to road center: ', distance_to_road_center)

            env.render()
            if done:
                if reward < 0:
                    print('*** CRASHED ***')
                print ('Final Reward = %.3f' % total_reward)
                break

if __name__ == '__main__':

    thread0 = threading.Thread(target=Particle.makeParticle)
    thread1 = threading.Thread(target=Particle.makeParticle)
    thread0.start()
    thread1.start()
    thread0.join()
    thread1.join()

