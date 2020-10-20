import cv2
import numpy as np
import math

from ctypes import POINTER
from collections import namedtuple

from scipy.spatial.transform import Rotation
from scipy.stats import truncnorm

from gym_duckietown.simulator import WINDOW_WIDTH, WINDOW_HEIGHT, NotInLane, get_dir_vec
from neural_perception.util.lane_extractor import detect_lane, display_lines
from pyglet import gl, window, image


RESIZE_IMG_SHAPE = (120, 160, 3)

OwnLanePosition0 = namedtuple('OwnLanePosition', 'dist dist_to_edge dot_dir angle_deg angle_rad')


class OwnLanePosition(OwnLanePosition0):
    def as_json_dict(self):
        """ Serialization-friendly format. """
        return dict(dist=self.dist,
                    dist_to_edge=self.dist_to_edge,
                    dot_dir=self.dot_dir,
                    angle_deg=self.angle_deg,
                    angle_rad=self.angle_rad)


def get_lane_pos(enviroment):
    pos = enviroment.cur_pos
    angle = enviroment.cur_angle
    point, tangent = enviroment.closest_curve_point(pos, angle)
    if point is None:
        msg = 'Point not in lane: %s' % pos
        raise NotInLane(msg)

    assert point is not None

    dirVec = get_dir_vec(angle)
    dotDir = np.dot(dirVec, tangent)
    dotDir = max(-1, min(1, dotDir))

    posVec = pos - point
    upVec = np.array([0, 1, 0])
    rightVec = np.cross(tangent, upVec)
    signedDist = np.dot(posVec, rightVec)
    dist_to_road_edge = 0.205 * enviroment.road_tile_size - signedDist
    angle_rad = math.acos(dotDir)

    if np.dot(dirVec, rightVec) < 0:
        angle_rad *= -1

    angle_deg = np.rad2deg(angle_rad)

    return OwnLanePosition(dist=signedDist,
                           dist_to_edge=dist_to_road_edge,
                           dot_dir=dotDir,
                           angle_deg=angle_deg,
                           angle_rad=angle_rad)


def own_render(environment, img):
    if environment.window is None:
        config = gl.Config(double_buffer=False)
        environment.window = window.Window(
            width=WINDOW_WIDTH,
            height=WINDOW_HEIGHT,
            resizable=False,
            config=config
        )

    environment.window.clear()
    environment.window.switch_to()
    environment.window.dispatch_events()
    gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
    gl.glMatrixMode(gl.GL_PROJECTION)
    gl.glLoadIdentity()
    gl.glMatrixMode(gl.GL_MODELVIEW)
    gl.glLoadIdentity()
    gl.glOrtho(0, WINDOW_WIDTH, 0, WINDOW_HEIGHT, 0, 10)
    width = img.shape[1]
    height = img.shape[0]
    img = np.ascontiguousarray(np.flip(img, axis=0))
    img_data = image.ImageData(
        width,
        height,
        'RGB',
        img.ctypes.data_as(POINTER(gl.GLubyte)),
        pitch=width * 3,
    )
    img_data.blit(
        0,
        0,
        0,
        width=WINDOW_WIDTH,
        height=WINDOW_HEIGHT
    )
    x, y, z = environment.cur_pos
    environment.text_label.text = "pos: (%.2f, %.2f, %.2f), angle: %d, steps: %d, speed: %.2f m/s" % (
        x, y, z,
        int(environment.cur_angle * 180 / math.pi),
        environment.step_count,
        environment.speed
    )
    environment.text_label.draw()

    gl.glFlush()


def plot_lanes(frame, environment, pos, angle, tangent, dist_error, error_frame=None):
    tangent /= np.linalg.norm(tangent)
    rot = Rotation.from_rotvec(np.radians(-90) * np.array([0, 1, 0]))
    rot_tangent = rot.apply(tangent * dist_error)
    new_pos = pos + rot_tangent[0]

    if error_frame is None:
        pos_save = environment.cur_pos
        angle_save = environment.cur_angle

        environment.cur_pos = new_pos
        environment.cur_angle = angle

        error_frame = environment.render(mode='rgb_array')

        environment.cur_pos = pos_save
        environment.cur_angle = angle_save

    lanes = detect_lane(error_frame)
    altered_frame = display_lines(frame, lanes)
    return altered_frame


def preprocess(frame):
    height, width, _ = RESIZE_IMG_SHAPE
    frame = cv2.resize(frame, (width, height))
    frame = frame[height // 3:, :]
    frame = frame / 255.
    return np.array([frame], dtype=np.float32)


def rot_y(deg):
    rad = np.deg2rad(deg)
    c = math.cos(rad)
    s = math.sin(rad)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])


def get_truncated_normal(mean=0.5, sd=1 / 4, low=0, upp=1):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


def get_mean_and_std(x, y):
    lo = int(np.min(x))
    hi = int(np.max(x))
    x_binned = np.arange(lo, hi+1)
    y_values_per_x = [[] for _ in range(len(x_binned))]

    for i in range(len(x)):
        val = int(x[i])
        val_e = y[i]
        y_values_per_x[val - lo].append(val_e)

    std_err = np.empty(shape=(len(y_values_per_x,)))
    for i, l in enumerate(y_values_per_x):
        if len(l) > 1:
            std_err[i] = np.std(l)
        else:
            std_err[i] = 0

    means = np.empty(shape=(len(y_values_per_x,)))
    for i, l in enumerate(y_values_per_x):
        if len(l) > 1:
            means[i] = np.mean(l)
        elif len(l) is 1:
            means[i] = l[0]
        else:
            means[i] = 0

    return x_binned, means, std_err


def lerp(v0, v1, t):
    return (1 - t) * v0 + t * v1


def angle_diff(a, b):
    c = a - b
    return (c + math.pi) % (math.pi * 2) - math.pi
