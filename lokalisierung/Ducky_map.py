import yaml
import random

ROW_MAP = 7.0
COLUMN_MAP = 8.0

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

class Particle:

    def __init__(self, p_x, p_y, weight, name, angle = -1):
        self.p_x = p_x
        self.p_y = p_y
        self.tile = None
        self.weight = weight
        self.name = name
        self.angle = angle

    def __repr__(self):
        return 'Particle '+str(self.name)

    def set_tile(self, map):
        self.tile = map.search_tile(int(self.p_x),int(self.p_y))

    def distance_to_wall(self):
        if self.tile.type == "straight/E" or self.tile.type == "straight/W":  # returns the distance to the closest wall (the wall can be above or under the particle)
            distance = self.p_y % 1
            if distance >= 0.5:
                return 1 - distance - self.tile.WhiteTapeWidth
            return distance - self.tile.WhiteTapeWidth
        if self.tile.type == "straight/N" or self.tile.type == "straight/S":  # returns the disctance to the closest wall (the wall can be on the left or on the right)
            distance = self.p_x % 1
            if distance >= 0.5:
                return 1 - distance
            return distance
        if self.tile.type == "3way_left/S":  # TODO distance to courve
            return self.p_x % 1
        if self.tile.type == "3way_left/N":
            return 1 - self.p_x % 1
        if self.tile.type == "3way_left/W":
            return self.p_y % 1
        if self.tile.type == "3way_left/E":
            return 1 - self.p_y % 1
        if self.tile.type == "curve_left/S":
            print("hi im a curve")


class Tile:

    def __init__(self, x, y, tile_type):
        self.x = x
        self.y = y
        self.type = tile_type
        self.WhiteTapeWidth = 0.048

    def __repr__(self):
        return 'Index:(' + str(self.x) + ', ' + str(self.y) + ') ' + 'Type: ' + self.type

    def index(self):
        return self.x, self.y



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


def filter_particles(particle_list, map:DuckieMap):
    for i in particle_list:
        # print(int(i.cordX))
        # print(int(i.cordY))
        wert = map.search_tile(int(i.p_x),int(i.p_y)).type
        #linex = tiles[0][int(i.cordY)]
        #wert = linex[int(i.cordX)]
        if wert == "grass" or wert == "asphalt" or wert == "floor":
            print(i)
            particle_list.remove(i)
    return particle_list

if __name__ == '__main__':
    my_map = DuckieMap("../gym_duckietown/maps/udem1.yaml")
    #print(my_map.tiles)
    particle_list = get_random_particles_list(10)
    for i in particle_list:
        i.set_tile(my_map)
    #print(particle_list)
    particle_list1 = filter_particles(particle_list, my_map)
    #print(particle_list)
    for i in particle_list1:
        print(f"particle name: {i.name} particle X: {i.p_x} particle y: {i.p_y} tile: {i.tile}")
        print(i.distance_to_wall())


