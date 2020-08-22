from lokalisierung.Particle import Particle
from lokalisierung.Ducky_map import ROW_MAP, COLUMN_MAP, DuckieMap
import functools
import random


class MCL:
    def __init__(self, p_number, map):
        self.p_number = p_number
        self.map = map
        self.p_list = None

    def spawn_particle_list(self):
        self.p_list = list()
        for i in range(0, self.p_number):
            randX = random.uniform(0.0, COLUMN_MAP)
            randY = random.uniform(0.0, ROW_MAP)
            rand_angle = random.uniform(0.0, 360)

            a_particle = Particle(randX, randY, 1, i, angle=rand_angle)
            a_particle.set_tile(self.map)
            self.p_list.append(a_particle)

    def weight_particle(self, action, distance_duckie, angle_duckie):
        for p in self.p_list:
            p.step(action)
            p.weight_calculator(distance_duckie, angle_duckie)

    def resampling(self):
        self.filter_particles()
        self.speichen_rad()

    def filter_particles(self):
        self.p_list = list(filter(lambda p: p.tile.type not in ['floor', 'asphalt', 'grass'], self.p_list))

    def speichen_rad(self):
        weight_sum = functools.reduce(lambda a, b: a.weight + b.weight, self.p_list)
        speichen_distanz = weight_sum / len(self.p_list)
        self.p_list = list(filter(lambda p: MCL.calculate(p, speichen_distanz), self.p_list))

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
