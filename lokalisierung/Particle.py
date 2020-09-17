from math import sqrt
import numpy as np
import threading

from lokalisierung import MCL
from threading import Thread
from gym_duckietown.simulator import _update_pos
from lokalisierung import Tile
from gym_duckietown.simulator import WHEEL_DIST, DEFAULT_FRAMERATE
from lokalisierung.Ducky_map import DuckieMap


class Particle:

    def __init__(self, p_x, p_y, weight, name, map, angle=-1):
        self.p_x = p_x
        self.p_y = p_y
        self.tile: Tile = None
        self.weight = weight
        self.name = name
        self.angle = angle
        self.tilesize = 0.61
        self.map = map

    def __repr__(self):
        return 'Particle ' + str(self.name) + str(self.tile)

    def set_tile(self):
        self.tile = self.map.search_tile(int(self.p_x), int(self.p_y))

    # todo: Jan should review this
    def step(self, action: np.ndarray):
        vel, angle = action
        baseline = WHEEL_DIST

        # assuming same motor constants k for both motors
        k_r = 27.0
        k_l = 27.0

        # adjusting k by gain and trim
        k_r_inv = (1.0 + 0) / k_r
        k_l_inv = (1.0 - 0) / k_l

        omega_r = (vel + 0.5 * angle * baseline) / 0.0318
        omega_l = (vel - 0.5 * angle * baseline) / 0.0318

        # conversion from motor rotation rate to duty cycle
        u_r = omega_r * k_r_inv
        u_l = omega_l * k_l_inv
        u_r_limited = max(min(u_r, 1.0), -1.0)
        u_l_limited = max(min(u_l, 1.0), -1.0)

        vels = np.array([u_l_limited, u_r_limited])
        vels = np.clip(vels, -1, 1)
        # Actions could be a Python list
        vels = np.array(vels)

        cur_pos = [self.p_x, 0.0, self.p_y]
        wheelVels = vels * vel * 1
        #print("wheelVels: ", wheelVels)
        old_angle = np.deg2rad(self.angle)
        new_pos, cur_angle = _update_pos(cur_pos,
                                         old_angle,
                                         WHEEL_DIST,
                                         wheelVels,
                                         deltaTime=1.0 / DEFAULT_FRAMERATE)
        self.angle = np.rad2deg(cur_angle)
        self.p_x = new_pos[0]
        self.p_y = new_pos[2]
        self.set_tile()
        return self.p_x, self.p_y, self.angle

    def distance_to_wall(self):
        px = self.p_x % 1
        py = self.p_y % 1
        p = np.array([px, py])
        #print("typ von tile", self.tile.type)
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
        if self.tile.type == "3way_left/S":
            if 180 > self.angle > 0:
                if py > 0.5:
                    eck = [0.562 / self.tilesize, 0.562 / self.tilesize]
                    return self.distance_from_point_to_point(eck, px, py)
                else:
                    eck = [0.562 / self.tilesize, 0.048 / self.tilesize]
                    return self.distance_from_point_to_point(eck, px, py)
            else:
                return self.p_x % 1
        if self.tile.type == "3way_left/N":
            if 360 > self.angle > 180:
                if py > 0.5:
                    eck = [0.048 / self.tilesize, 0.562 / self.tilesize]
                    return self.distance_from_point_to_point(eck, px, py)
                else:
                    eck = [0.048 / self.tilesize, 0.048 / self.tilesize]
                    return self.distance_from_point_to_point(eck, px, py)
            else:
                return 1 - self.p_x % 1
        if self.tile.type == "3way_left/W":
            if self.angle < 90 or self.angle > 270:
                if px < 0.5:
                    eck = [0.048 / self.tilesize, 0.562 / self.tilesize]
                    return self.distance_from_point_to_point(eck, px, py)
                else:
                    eck = [0.562 / self.tilesize, 0.562 / self.tilesize]
                    return self.distance_from_point_to_point(eck, px, py)
            else:
                return self.p_y % 1
        if self.tile.type == "3way_left/E":
            if 90 < self.angle < 270:
                if px < 0.5:
                    eck = [0.048 / self.tilesize, 0.048 / self.tilesize]
                    return self.distance_from_point_to_point(eck, px, py)
                else:
                    eck = [0.562 / self.tilesize, 0.048 / self.tilesize]
                    return self.distance_from_point_to_point(eck, px, py)
            else:
                return 1 - self.p_y % 1
        if self.tile.type == "curve_left/S":
            print("hi im a curve")
            if 90 < self.angle < 180:
                p1 = np.array([0.483 / self.tilesize, 0.02 / self.tilesize])
                p2 = np.array([0.53 / self.tilesize, 0.08 / self.tilesize])
                distance = self.distance_from_point_to_lines(p, p1, p2)
                return distance
            else:
                circle_centre = [0.61 / self.tilesize, 0]
                return self.dist_to_circle(circle_centre[0], circle_centre[1], px, py, 0.51 / self.tilesize)
        if self.tile.type in ['curve_left/W', 'curve_right/N']:
            if 0 < self.angle < 180:
                p1 = np.array([0.61 / self.tilesize, 0.483 / self.tilesize])
                p2 = np.array([0.53 / self.tilesize, 0.61 / self.tilesize])
                return self.distance_from_point_to_lines(p, p1, p2)
            else:
                circle_centre = [0.61 / self.tilesize, 0.61 / self.tilesize]
                return self.dist_to_circle(circle_centre[0], circle_centre[1], px, py, 0.51 / self.tilesize)
        if self.tile.type == 'curve_left/E':
            if 180 < self.angle < 360:
                p1 = np.array([0.027 / self.tilesize, 0])
                p2 = np.array([0, 0.027 / self.tilesize])
                return self.distance_from_point_to_lines(p, p1, p2)
            else:
                circle_centre = [0, 0]
                return self.dist_to_circle(circle_centre[0], circle_centre[1], px, py, 0.51 / self.tilesize)
        if self.tile.type == 'curve_left/N':
            if 180 < self.angle < 360:
                p1 = np.array([0, 0.483 / self.tilesize])
                p2 = np.array([0.027 / self.tilesize, 0.61 / self.tilesize])
                return self.distance_from_point_to_lines(p, p1, p2)
            else:
                circle_centre = [0, 0.61 / self.tilesize]
                return self.dist_to_circle(circle_centre[0], circle_centre[1], px, py, 0.51 / self.tilesize)
        print("no return")
    @staticmethod
    def distance_from_point_to_lines(p3, p1, p2):
        return np.linalg.norm(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)

    @staticmethod
    def distance_from_point_to_point(eck, px, py):
        return sqrt((eck[0] - px) ** 2 + (eck[1] - py) ** 2)

    # Function to find the shortest distance
    @staticmethod
    def dist_to_circle(x1, y1, x2, y2, r):
        return np.abs(((((x2 - x1) ** 2) + ((y2 - y1) ** 2)) ** (1 / 2)) - r)

    def angle_to_wall_straight(self):
        if self.tile.type == "straight/E" or self.tile.type == "straight/W":
            if self.direction() == 'NE':
                return self.angle
            if self.direction() == 'SE':
                return self.angle - 360
            if self.direction() == 'NW':
                return self.angle - 180
            if self.direction() == 'SW':
                return self.angle % 90
        if self.tile.type == "straight/N" or self.tile.type == "straight/S":
            if self.direction() == 'NE':
                return self.angle - 90
            if self.direction() == 'NW':
                return self.angle % 90
            if self.direction() == 'SW':
                return self.angle - 270
            if self.direction() == 'SE':
                return self.angle % 90

    def angle_to_wall_3way(self):
        if self.tile.type == "3way_left/S":
            if self.direction() == 'SW':
                return self.angle - 270
            if self.direction() == 'SE':
                return self.angle % 90
            return 'nothing'
        if self.tile.type == "3way_left/N":
            if self.direction() == 'NE':
                return self.angle - 90
            if self.direction() == 'NW':
                return self.angle % 90
            return 'nothing'
        if self.tile.type == "3way_left/W":
            if self.direction() == 'NW':
                return self.angle - 180
            if self.direction() == 'SW':
                return self.angle % 90
            return 'nothing'
        if self.tile.type == "3way_left/E":
            if self.direction() == 'NE':
                return self.angle
            if self.direction() == 'SE':
                return self.angle - 360
            return 'nothing'

    def angle_to_wall(self):
        if self.tile.type == "straight/E" or self.tile.type == "straight/W" or self.tile.type == "straight/N" or self.tile.type == "straight/S":
            return self.angle_to_wall_straight()
        if self.tile.type == "3way_left/S" or self.tile.type == "3way_left/N" or self.tile.type == "3way_left/W" or self.tile.type == "3way_left/E" or self.tile.type == "curve_left/S":
            return self.angle_to_wall_3way()
        if self.tile.type.startswith('curve_left') or self.tile.type.startswith('curve_right'):
            return self.angle_to_wall_curve()
        print("no return in agnle to wall")
    def angle_to_wall_curve(self):
        if self.tile.type in ['curve_left/W', 'curve_right/N']:
            if self.direction() in ['NE', 'NW']:
                value = (1 - (self.p_x - self.p_x // 1)) / (1 - (self.p_y - self.p_y // 1))
                angle_of_tangent = np.arctan([value])
                return self.angle - np.rad2deg(angle_of_tangent[0])
            if self.direction() in ['SE', 'SW']:
                return self.angle - 45
        if self.tile.type == 'curve_left/E':
            if self.direction() in ['NE', 'NW']:
                value = (self.p_x - self.p_x // 1) / (self.p_y - self.p_y // 1)
                angle_of_tangent = np.arctan([value])
                return self.angle - np.rad2deg(angle_of_tangent[0])
            if self.direction() in ['SE', 'SW']:
                return self.angle - 45
        if self.tile.type == 'curve_left/S':
            if self.direction() in ['NE', 'NW']:
                return self.angle - 45
            if self.direction() in ['SE', 'SW']:
                value = (1 - (self.p_x - self.p_x // 1)) / (self.p_y - self.p_y // 1)
                angle_of_tangent = np.arctan([value])
                return 360 - self.angle - np.rad2deg(angle_of_tangent[0])
        if self.tile.type == 'curve_left/N':
            if self.direction() in ['NE', 'NW']:
                value = (self.p_x - self.p_x // 1) / (1 - (self.p_y - self.p_y // 1))
                angle_of_tangent = np.arctan([value])
                return self.angle - (180 - np.rad2deg(angle_of_tangent[0]))
            if self.direction() in ['SE', 'SW']:
                return self.angle - 135

    def direction(self):
        if self.angle < 90:
            return 'NE'
        if self.angle < 180:
            return 'NW'
        if self.angle < 270:
            return 'SW'
        if self.angle < 360:
            return 'SE'

    def weight_calculator(self, distance, angle):
        #print("dtw =", self.distance_to_wall())
        #print("distance ", distance)
        #self.weight = (self.distance_to_wall() - distance) / distance
        self.weight = abs(1 * ((self.distance_to_wall() - distance) / distance)) #* ((self.angle_to_wall() - angle) / angle))
        return self.weight

if __name__ == '__main__':
    print("hello world")
