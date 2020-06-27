import random

import yaml

from lokalisierung.Particle import Particle
from lokalisierung.Tile import Tile


class DuckieMap:

    def __init__(self, yaml_file):
        with open(yaml_file, "r") as stream:
            try:
                out = yaml.safe_load(stream)
                tiles = out["tiles"]
                w = len(tiles[0])
                h = len(tiles)
                self.tiles = [[0 for x in range(h)] for y in range(w)]
                for i in range(h):
                    for j in range(w):
                        self.tiles[j][i] = Tile(j, i, tiles[i][j])
            except yaml.YAMLError as exc:
                print(exc)

    def search_tile(self, x, y):
        return self.tiles[x][y]



def get_random_particles_list(count):
    p_list = list()
    i = 0
    while i < count:
        randX = random.uniform(0.0, 8.0)
        randY = random.uniform(0.0, 7.0)

        a_particle = Particle(randX, randY, 0, i)

        p_list.append(a_particle)
        i += 1
    return p_list


def filter_particles(particles_list, a_map: DuckieMap):
    return list(filter(lambda i: not filter_condition(a_map.search_tile(int(i.p_x), int(i.p_y)).type), particles_list))


def filter_condition(wert):
    return wert in ['grass', 'asphalt', 'floor']


if __name__ == '__main__':
    my_map = DuckieMap("../gym_duckietown/maps/udem1.yaml")
    # print(my_map.tiles)
    particle_list = get_random_particles_list(10)
    for i in particle_list:
        i.set_tile(my_map)
    # print(particle_list)
    particle_list1 = filter_particles(particle_list, my_map)
    # print(particle_list)
    for i in particle_list1:
        print(f"particle name: {i.name} particle X: {i.p_x} particle y: {i.p_y} tile: {i.tile}")
        print(i.distance_to_wall())
