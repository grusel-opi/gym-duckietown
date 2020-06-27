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
