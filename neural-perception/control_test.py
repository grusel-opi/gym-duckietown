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
from lane_extractor import detect_lane, display_lines
from data_generator import OwnLanePosition
from gym_duckietown.simulator import WINDOW_WIDTH, WINDOW_HEIGHT, NotInLane, get_dir_vec
from gym_duckietown.envs import DuckietownEnv
import tensorflow as tf
from pyglet import gl, window, image


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

RESIZE_IMG_SHAPE = (120, 160, 3)


def get_lane_pos3(environment):
    pos = environment.cur_pos
    angle = environment.cur_angle
    point, tangent = environment.closest_curve_point(pos, angle)
    if point is None:
        msg = 'Point not in lane: %s' % pos
        raise NotInLane(msg)

    assert point is not None

    track_width = 0.4
    a = track_width / 2
    rot = Rotation.from_rotvec(np.radians(-90) * np.array([0, 1, 0]))
    rot_tangent = rot.apply(tangent * a)
    new_point = point + rot_tangent

    dir_vec = get_dir_vec(angle)
    dot_dir = np.dot(dir_vec, tangent)
    dot_dir = max(-1, min(1, dot_dir))

    # Compute the signed distance to the curve
    # Right of the curve is negative, left is positive
    new_pos_vec = new_point - pos
    pos_vec = pos - point
    up_vec = np.array([0, 1, 0])
    right_vec = np.cross(tangent, up_vec)
    signed_dist_egde = np.dot(new_pos_vec, right_vec)
    signed_dist_center = np.dot(pos_vec, right_vec)

    # Compute the signed angle between the direction and curve tangent
    # Right of the tangent is negative, left is positive
    angle_rad = math.acos(dot_dir)

    if np.dot(dir_vec, right_vec) < 0:
        angle_rad *= -1

    angle_deg = np.rad2deg(angle_rad)

    return OwnLanePosition(dist=signed_dist_center, dist_to_edge=signed_dist_egde, dot_dir=dot_dir,
                           angle_deg=angle_deg,
                           angle_rad=angle_rad)


def display_custom_image(environment, img):

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

    # Bind the default frame buffer
    gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)

    # Setup orghogonal projection
    gl.glMatrixMode(gl.GL_PROJECTION)
    gl.glLoadIdentity()
    gl.glMatrixMode(gl.GL_MODELVIEW)
    gl.glLoadIdentity()
    gl.glOrtho(0, WINDOW_WIDTH, 0, WINDOW_HEIGHT, 0, 10)

    # Draw the image to the rendering window
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

    # Display position/state information
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
    frame = frame[0:height // 2, :]
    frame = frame / 255.
    return np.array([frame])


if __name__ == '__main__':
    env = DuckietownEnv()
    obs = env.reset()

    obs = preprocess(obs)

    env.render()
    total_reward = 0
    # model = tf.keras.models.load_model('./saved_model/24.06.2020-13:39:50/')

    k_p = 10
    k_d = 1
    speed = 0.2

    while True:

        lane_pose = get_lane_pos3(env)
        distance_to_road_edge = lane_pose.dist_to_edge * 100  # using 100th of tile size
        distance_to_road_center = lane_pose.dist
        angle_from_straight_in_rads = lane_pose.angle_rad

        # d = model.predict(obs)[0]

        # dist_err = distance_to_road_edge - d

        # print()
        # print(d)
        # print(distance_to_road_edge)
        # print("error: {}".format(dist_err))

        steering = k_p * distance_to_road_center + k_d * angle_from_straight_in_rads

        obs, reward, done, info = env.step(np.array([speed, steering]))
        total_reward += reward
        obs = preprocess(obs)

        _, t = env.closest_curve_point(env.cur_pos, env.cur_angle)

        rendered = env.render(mode='rgb_array')
        # rendered = plot_lanes(rendered, env, env.cur_pos, env.cur_angle, t, dist_err)
        rendered = plot_lanes(rendered, env, env.cur_pos, env.cur_angle, t, 0, error_frame=rendered)
        display_custom_image(env, rendered)

        if done:
            if reward < 0:
                print('*** CRASHED ***')
            print('Final Reward = %.3f' % total_reward)
            break
