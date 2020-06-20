import yaml


class DuckieMap:

    def __init__(self, yaml_file):
        with open(yaml_file, "r") as stream:
            try:
                out = yaml.safe_load(stream)
                tiles = out["tiles"]
                w = len(tiles[0])
                h = len(tiles)
                self.tiles = [[0 for x in range(w)] for y in range(h)]
                for i in range(len(tiles)):
                    for j in range(len(tiles[0])):
                        self.tiles[i][j] = Tile(i, j, tiles[i][j])
            except yaml.YAMLError as exc:
                print(exc)

    def search_tile(self, x, y):
        return self.tiles[x][y]


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

    def distance_to_wall(self, p_x, p_y):
        if self.type == "straight/E" or self.type == "straight/W":  # returns the distance to the closest wall (the wall can be above or under the particle)
            distance = p_y % 1
            if distance >= 0.5:
                return 1 - distance - self.WhiteTapeWidh
            return distance - self.WhiteTapeWidh
        if self.type == "straight/N" or self.type == "straight/S":  # returns the disctance to the closest wall (the wall can be on the left or on the right)
            distance = p_x % 1
            if distance >= 0.5:
                return 1 - distance
            return distance
        elif self.type == "3way_left/N":  # TODO distance to courve
            print("3way_left/N")
            return None
        print("should be distance to wall in courve")
        return None


if __name__ == '__main__':
    my_map = DuckieMap("../gym_duckietown/maps/udem1.yaml")
    print(my_map.tiles)

