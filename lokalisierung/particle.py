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


class Particle():

    def __init__(self, env:DuckietownEnv):
        self.env = env
        self.env.reset()
        self.start_pos = self.env.cur_pos
        self.dist_to_centre = None
        self.angle_to_centre = None
        self.weight = None

    def step(self, aktion):
        obs, reward, done, info = self.env.step(aktion)
        if not done:
            lane_pose = self.env.get_lane_pos2(self.env.cur_pos, self.env.cur_angle)
            self.dist_to_centre = lane_pose.dist
            self.angle_to_centre = lane_pose.angle_rad
        return done

    def weight_to_robot(self, dist_to_centre):
        self.weight = 1 - (abs(dist_to_centre - self.dist_to_centre) / dist_to_centre)


def get_random_particles_list(count):

    p_list = list()
    i = 0
    while i < count:
        a_particle = Particle(DuckietownEnv(
            map_name=args.map_name,
            domain_rand=False,
            draw_bbox=False
        ))
        p_list.append(a_particle)
        i += 1

    return p_list

def paint_weight_graph(x_array=None,y_array=None,weight=None):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    x = x_array
    y = y_array

    X, Y = np.meshgrid(x, y)
    Z = weight

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.contour3D(X, Y, Z, 50, cmap='binary')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('weight')
    plt.show()

if __name__ == '__main__':

    # thread0 = threading.Thread(target=Particle.makeParticle)
    # thread1 = threading.Thread(target=Particle.makeParticle)
    # thread0.start()
    # thread1.start()
    # thread0.join()
    # thread1.join()

    a_list = (get_random_particles_list(10))
    x_array =[]
    y_array = []
    z_array = []
    print(a_list)
    speed = 0.2
    steering = 0
    i = 0
    while i < 200:
        for p in a_list:
            x_array.append(p.start_pos[0])
            y_array.append(p.start_pos[2])
            if p.step([speed, steering]):
                print("Particle ", a_list.index(p), "is dead")
                a_list.remove(p)
            else:
                p.weight_to_robot(2.0)
                z_array.append(p.weight)
                print("particle ", a_list.index(p), " has weight ", p.weight)
        i += 1
        #paint_weight_graph(x_array,y_array,z_array)