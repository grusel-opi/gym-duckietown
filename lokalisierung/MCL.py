from lokalisierung.Particle import Particle
from lokalisierung.Ducky_map import DuckieMap
import random
from bisect import bisect_left
import numpy as np


class MCL:
    def __init__(self, p_number, map, env):
        self.p_number = p_number
        self.map = map
        self.p_list = None
        self.env = env

    def spawn_particle_list(self, robot_pos, robot_angle):
        self.p_list = list()
        p_x, p_y = (robot_pos[0], robot_pos[2])
        i = 0
        while i < self.p_number:
            randX = random.uniform(p_x - 0.25, p_x//1 + 0.25)
            randY = random.uniform(p_y - 0.25, p_y // 1 + 0.25)
            rand_angle = random.uniform(robot_angle - np.deg2rad(15), robot_angle + np.deg2rad(15))

            a_particle = Particle(randX, randY, 1, i, self.map,self.env, angle=rand_angle)
            a_particle.set_tile()
            if a_particle.tile.type not in ['floor', 'asphalt', 'grass']:
                self.p_list.append(a_particle)
                i = i + 1

    def integrate_movement(self, action):
        for p in self.p_list:
            p.step(action)
        self.filter_particles()

    def integrate_measurement(self, distance_duckie, angle_duckie):
        for p in self.p_list:
            p.weight_calculator(distance_duckie,angle_duckie)

    def resampling(self):
        arr_particles = []
        for i in range(0, len(self.p_list)):
            idx = self.roulette_rad()
            arr_particles.append(self.p_list[idx])

        sum_py = 0
        sum_px = 0
        sum_angle = 0
        for x in arr_particles:
            sum_px = sum_px + x.p_x
            sum_py = sum_py + x.p_y
            sum_angle += x.angle
        # sum_px = functools.reduce(lambda a,b : a.p_x + b.p_x, arr_chosenones)
        # sum_py = functools.reduce(lambda a,b : a.p_y + b.p_y, arr_chosenones)
        possible_location = [sum_px / len(arr_particles), 0, sum_py / len(arr_particles)]
        possible_angle = sum_angle / len(arr_particles)
        return arr_particles, possible_location, possible_angle

    def filter_particles(self):
        self.p_list = list(filter(lambda p: p.tile.type not in ['floor', 'asphalt', 'grass'], self.p_list))

    def roulette_rad(self):
        weight_arr = []
        weight_of_particle = 0
        for p in self.p_list:
            weight_of_particle += p.weight
            weight_arr.append(weight_of_particle)
        the_chosen_one = random.uniform(0, weight_of_particle)
        idx_particle = bisect_left(weight_arr, the_chosen_one)
        return idx_particle

    def weight_reset(self):
        for p in self.p_list:
            p.weight = 1


if __name__ == '__main__':
    my_map = DuckieMap("../gym_duckietown/maps/udem1.yaml")
    MCL = MCL(10, my_map)
    MCL.spawn_particle_list()
    print(MCL.p_list)
    MCL.filter_particles()
    print(MCL.p_list)
