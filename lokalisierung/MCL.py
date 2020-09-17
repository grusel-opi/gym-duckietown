from lokalisierung.Particle import Particle
from lokalisierung.Ducky_map import ROW_MAP, COLUMN_MAP, DuckieMap
import functools
import random
from bisect import bisect_right, bisect_left


class MCL:
    def __init__(self, p_number, map):
        self.p_number = p_number
        self.map = map
        self.p_list = None

    def spawn_particle_list(self):
        self.p_list = list()
        i = 0
        while(i < self.p_number):
            randX = random.uniform(0.0, COLUMN_MAP)
            randY = random.uniform(0.0, ROW_MAP)
            rand_angle = random.uniform(0.0, 360)

            a_particle = Particle(randX, randY, 1, i, self.map, angle=rand_angle)
            a_particle.set_tile()
            if a_particle.tile.type not in ['floor', 'asphalt', 'grass']:
                self.p_list.append(a_particle)
                i = i + 1

    def weight_particles(self, action, distance_duckie, angle_duckie):
        for p in self.p_list:
            p.step(action)
        self.filter_particles()
        for p in self.p_list:
            p.weight_calculator(distance_duckie, angle_duckie)

    def resampling(self):
        self.roulette_rad()
        arr_particles = []
        for i in range(0, len(self.p_list)):
            idx = self.roulette_rad()
            # print("idx =", idx)
            # print("len p_list", len(self.p_list))
            arr_particles.append(self.p_list[idx])
        return arr_particles

    def filter_particles(self):
        self.p_list = list(filter(lambda p: p.tile.type not in ['floor', 'asphalt', 'grass'], self.p_list))

    def roulette_rad(self):
        weight_arr = []
        weight_of_particle = 0
        for p in self.p_list:
            weight_of_particle += p.weight
            weight_arr.append(weight_of_particle)

        the_chosen_one = random.uniform(0, weight_of_particle)
        #print("chosen and weight of particle", the_chosen_one, weight_of_particle)
        idx_particle = bisect_left(weight_arr, the_chosen_one)
        return idx_particle


    @staticmethod
    def calculate(particle, speichendistanz):
        if ((particle.weight // speichendistanz) + 1) is 1:
            return False
        return True


if __name__ == '__main__':
    my_map = DuckieMap("../gym_duckietown/maps/udem1.yaml")
    MCL = MCL(10, my_map)
    MCL.spawn_particle_list()
    print(MCL.p_list)
    MCL.filter_particles()
    print(MCL.p_list)
