#!/usr/bin/env python3

"""
Simple exercise to construct a controller that controls the simulated Duckiebot using pose.
"""

import cv2
import numpy as np
import math
import os
from ctypes import POINTER
from scipy.spatial.transform import Rotation
from neural_perception.lane_extractor import detect_lane, display_lines
from neural_perception.data_generator import OwnLanePosition
from gym_duckietown.simulator import WINDOW_WIDTH, WINDOW_HEIGHT, NotInLane, get_dir_vec
from gym_duckietown.envs import DuckietownEnv
import tensorflow as tf
from pyglet import gl, window, image
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

RESIZE_IMG_SHAPE = (120, 160, 3)


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
    dist_to_road_edge = 0.25 * enviroment.road_tile_size - signedDist
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

    # Force execution of queued commands
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


if __name__ == '__main__':
    env = DuckietownEnv(domain_rand=False,
                        draw_bbox=False)

    obs = env.reset()

    obs = preprocess(obs)

    env.render()

    model = tf.keras.models.load_model('../../lean-test/saved_model/18.09.2020-13:24:04/')

    k_p = 1
    k_d = 1
    speed = 0.2

    steps = env.max_steps = 10_000

    for i in range(steps):

        y_hat = model(obs)

        d_hat = y_hat[0][0]
        a_hat = y_hat[0][1]

        distance_to_road_center = (d_hat - 25.) / 100.
        angle_from_straight_in_rad = (a_hat*2*np.pi) / 360.

        steering = k_p * distance_to_road_center + k_d * angle_from_straight_in_rad
        command = np.array([speed, steering])
        obs, _, done, _ = env.step(command)

        obs = preprocess(obs)
        env.render()

        if done:
            env.reset()
