# import time
# import sys
# import argparse
# import math
# import numpy as np
# import yaml
# import gym
# import random
# from gym_duckietown.envs import DuckietownEnv
# import threading
#
# # include <X11/Xlib.h>
#
#
# parser = argparse.ArgumentParser()
# parser.add_argument('--env-name', default=None)
# parser.add_argument('--map-name', default='udem1')
# parser.add_argument('--no-pause', action='store_true', help="don't pause on failure")
# args = parser.parse_args()
#
# tiles = []
#
#
# class Particle():
#
#     def __init__(self, env: DuckietownEnv):
#         self.env = env
#         self.env.reset()
#         self.start_pos = self.env.cur_pos
#         self.dist_to_centre = None
#         self.angle_to_centre = None
#         self.weight = None
#
#     def step(self, aktion):#TODO remake step function for the new particle
#         obs, reward, done, info = self.env.step(aktion)
#         if not done:
#             lane_pose = self.env.get_lane_pos2(self.env.cur_pos, self.env.cur_angle)
#             self.dist_to_centre = lane_pose.dist
#             self.angle_to_centre = lane_pose.angle_rad
#         return done
#
#     def weight_to_robot(self, dist_to_centre):
#         self.weight = 1 - (abs(dist_to_centre - self.dist_to_centre) / dist_to_centre)
#
#
# def get_random_particles_list(count, angle):
#     p_list = list()
#     i = 0
#     while i < count:
#         randX = random.uniform(0.0, 8.0)
#         randY = random.uniform(0.0, 7.0)
#
#         a_particle = New_Particle(randX, randY, angle, 0, i)
#
#         p_list.append(a_particle)
#         i += 1
#
#     return p_list
#
#
# def paint_weight_graph(x_array=None, y_array=None, weight=None):#TODO remake for the new particle
#     import matplotlib.pyplot as plt
#     from mpl_toolkits.mplot3d import Axes3D
#     x = x_array
#     y = y_array
#
#     X, Y = np.meshgrid(x, y)
#     Z = weight
#
#     fig = plt.figure()
#     ax = plt.axes(projection='3d')
#     ax.contour3D(X, Y, Z, 50, cmap='binary')
#     ax.set_xlabel('x')
#     ax.set_ylabel('y')
#     ax.set_zlabel('weight')
#     plt.show()
#
#
# x = []
#
#
# def filter_particles(particle_list):
#     for i in particle_list:
#         # print(int(i.cordX))
#         # print(int(i.cordY))
#         linex = tiles[0][int(i.cordY)]
#         wert = linex[int(i.cordX)]
#         if wert == "grass" or wert == "asphalt" or wert == "floor":
#             particle_list.remove(i)
#     return particle_list
#
# def distance_to_wall(particle):
#     linex = tiles[0][int(particle.cordY)]
#     wert = linex[int(particle.cordX)]
#
#     if wert == "straight/E" or wert == "straight/W": #returns the distance to the closest wall (the wall can be above or under the particle)
#         if wert == "straight/E":
#             print("straight/E")
#         else: print("straight/W")
#         distance = particle.cordY % 1
#         if distance >= 0.5:
#             return 1 - distance
#         return distance
#     if wert == "straight/N" or wert == "straight/S": #returns the disctance to the closest wall (the wall can be on the left or on the right)
#         distance = particle.cordX % 1
#         if distance >= 0.5:
#             return 1 - distance
#         return distance
#     elif wert == "3way_left/N": #TODO distance to courve
#         print("3way_left/N")
#         return None
#     print("should be distance to wall in courve")
#     return None
#
#
# # --------------------------------------------------------------------
# class New_Particle():
#
#     def __init__(self, cordX, cordY, angle, weight, name):
#         self.name = name
#         self.cordX = cordX
#         self.cordY = cordY
#         self.angle = angle
#         self.weight = weight
#
#     with open("../gym_duckietown/maps/udem1.yaml", "r") as stream:
#         try:
#             out = yaml.safe_load(stream)
#             tile = out["tiles"]
#             tiles.append(tile)
#         except yaml.YAMLError as exc:
#             print(exc)
#
#
# # ----------------------------------------------------------------------
# if __name__ == '__main__':
#     # thread0 = threading.Thread(target=Particle.makeParticle)
#     # thread1 = threading.Thread(target=Particle.makeParticle)
#     # thread0.start()
#     # thread1.start()
#     # thread0.join()
#     # thread1.join()
#     #
#     a_list = (get_random_particles_list(1000, 70))
#     # print(a_list)
#     # x_array =[]
#     # y_array = []
#     # z_array = []
#     # print(a_list)
#     # speed = 0.2
#     # steering = 0
#     # i = 0
#     # while i < 200:
#     #     for p in a_list:
#     #         x_array.append(p.start_pos[0])
#     #         y_array.append(p.start_pos[2])
#     #         if p.step([speed, steering]):
#     #             print("Particle ", a_list.index(p), "is dead")
#     #             a_list.remove(p)
#     #         else:
#     #             p.weight_to_robot(2.0)
#     #             z_array.append(p.weight)
#     #             print("particle ", a_list.index(p), " has weight ", p.weight, " has postion", p.env.cur_pos)
#     #     i += 1
#     # paint_weight_graph(x_array,y_array,z_array)
#
#     # print(tiles)
#
#     # for i in a_list:
#     #    print("X cordinate: ",i.cordX, "Y cordiante: ",i.cordY)
#     for i in a_list:
#         print(f"Particle: {i.name} X: {i.cordX} Y: {i.cordY}")
#     print(len(a_list))
#         # print(wert)
#     a_list = filter_particles(a_list)
#     print(len(a_list))
#     for i in a_list:
#         print(f"Particle: {i.name} X: {i.cordX} Y: {i.cordY} distance: {distance_to_wall(i)}")
