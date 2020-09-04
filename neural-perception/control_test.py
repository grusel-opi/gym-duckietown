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
    return np.array([frame])


if __name__ == '__main__':
    env = DuckietownEnv(domain_rand=False,
                        draw_bbox=False)

    obs = env.reset()

    # env.cam_height = env.cam_height*1.5

    obs = preprocess(obs)

    # env.render()

    total_reward = 0
    model = tf.keras.models.load_model('../../lean-test/saved_model/04.09.2020-00:44:59/')

    k_p = 1
    k_d = 1
    speed = 0.2

    steps = env.max_steps = 1_000
    period = 2

    visual = False

    all_errors = []
    straight_tile_errors = []
    curve_left_errors = []
    curve_right_errors = []
    three_way_errors = []
    four_way_errors = []
    distances = []
    angles = []

    for i in range(steps):

        lane_pose = get_lane_pos(env)
        distance_to_road_edge = lane_pose.dist_to_edge*100
        distance_to_road_center = lane_pose.dist
        angle_from_straight_in_deg = lane_pose.angle_rad * 360/(2*np.pi)

        if i % period == 0:
            percent = i * period * 100 / steps
            print(f'\rrunning: {percent} %', end='\r')

            d = model.predict(obs)[0][0]

            dist_err = abs(distance_to_road_edge - d)
            all_errors.append(dist_err)
            distances.append(distance_to_road_edge)
            angles.append(angle_from_straight_in_deg)

            kind = env._get_tile(env.cur_pos[0], env.cur_pos[2])['kind']

            if kind.startswith('straight'):
                straight_tile_errors.append(dist_err)
            elif kind == 'curve_left':
                curve_left_errors.append(dist_err)
            elif kind == 'curve_right':
                curve_right_errors.append(dist_err)
            elif kind.startswith('3way'):
                three_way_errors.append(dist_err)
            elif kind.startswith('4way'):
                four_way_errors.append(dist_err)

            if visual:
                print()
                print("actu a:", angle_from_straight_in_deg)
                print("actu d:", distance_to_road_edge)
                print("pred d:", d)
                print("error:  {}".format(dist_err))

        steps += 1

        steering = k_p * distance_to_road_center + k_d * angle_from_straight_in_deg

        obs, _, done, _ = env.step(np.array([speed, steering]))

        obs = preprocess(obs)

        # if visual:
        #     _, t = env.closest_curve_point(env.cur_pos, env.cur_angle)
        #     rendered = env.render(mode='rgb_array')
        #     rendered = plot_lanes(rendered, env, env.cur_pos, env.cur_angle, t, dist_err)
        #     rendered = plot_lanes(rendered, env, env.cur_pos, env.cur_angle, t, 0, error_frame=rendered)
        #     display_custom_image(env, rendered)

        if done:
            env.reset()

    x = np.arange(len(all_errors))

    print()
    print("stats:")
    print()
    print("error mean: ", np.mean(all_errors))
    print("straight error mean: {}".format(np.mean(straight_tile_errors)))
    print("curve left error mean: {}".format(np.mean(curve_left_errors)))
    print("curve right error mean: {}".format(np.mean(curve_right_errors)))
    print("three_way error mean: {}".format(np.mean(three_way_errors)))
    print("four_way error mean: {}".format(np.mean(four_way_errors)))

    plt.figure()
    plt.title("hist")
    plt.hist(distances)

    plt.figure()
    plt.title("scatter")
    plt.scatter(distances, all_errors)

    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('step')
    ax1.set_ylabel('error', color=color)
    ax1.plot(x, all_errors, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()

    color = 'tab:blue'
    ax2.set_ylabel('distances', color=color)
    ax2.plot(x, distances, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    ax3 = ax1.twinx()

    color = 'tab:green'
    ax3.set_ylabel('angle from straight', color=color)
    ax3.plot(x, angles, color=color)
    ax3.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.show()

