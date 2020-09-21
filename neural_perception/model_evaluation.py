#!/usr/bin/env python3

"""
Simple exercise to construct a controller that controls the simulated Duckiebot using pose.
"""

import numpy as np
import os
from gym_duckietown.envs import DuckietownEnv
import tensorflow as tf
import matplotlib.pyplot as plt

from neural_perception.util import preprocess, get_lane_pos, own_render

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


env = DuckietownEnv(domain_rand=False,
                    draw_bbox=False)

obs = env.reset()
obs = preprocess(obs)

env.render()

model = tf.keras.models.load_model('../../lean-test/saved_model/17.09.2020-13:40:11/')

k_p = 1
k_d = 1
speed = 0.2

steps = env.max_steps = 1_000

visual = True

errors = []
straight_tile_errors = []
curve_left_errors = []
curve_right_errors = []
three_way_errors = []
four_way_errors = []

labels = []

for i in range(steps):

    percent = round(i * 100 / steps, 2)
    print(f'\rrunning: {percent} %', end='\r')

    lane_pose = get_lane_pos(env)

    distance_to_road_edge = lane_pose.dist_to_edge * 100
    distance_to_road_center = lane_pose.dist
    angle_from_straight_in_rad = lane_pose.angle_rad
    angle_from_straight_in_deg = lane_pose.angle_deg

    y_hat = model(obs)
    d = y_hat[0][0]
    a = y_hat[0][1]

    dist_err = abs(distance_to_road_edge - d)
    angle_err = abs(angle_from_straight_in_deg - a)

    errors.append((dist_err, angle_err))
    labels.append((distance_to_road_edge, angle_from_straight_in_deg))

    kind = env._get_tile(env.cur_pos[0], env.cur_pos[2])['kind']

    if kind.startswith('straight'):
        straight_tile_errors.append((dist_err, angle_err))
    elif kind == 'curve_left':
        curve_left_errors.append((dist_err, angle_err))
    elif kind == 'curve_right':
        curve_right_errors.append((dist_err, angle_err))
    elif kind.startswith('3way'):
        three_way_errors.append((dist_err, angle_err))
    elif kind.startswith('4way'):
        four_way_errors.append((dist_err, angle_err))

    if visual:
        print()
        print("actu: {}, {}".format(distance_to_road_edge, angle_from_straight_in_deg))
        print("pred: {}, {}".format(d, a))
        print("error: {}, {}".format(dist_err, angle_err))

    steering = k_p * distance_to_road_center + k_d * angle_from_straight_in_rad
    command = np.array([speed, steering])
    obs, _, done, _ = env.step(command)

    obs = preprocess(obs)

    if visual:
        rendered = env.render(mode='rgb_array')
        # _, t = env.closest_curve_point(env.cur_pos, env.cur_angle)
        # rendered = plot_lanes(rendered, env, env.cur_pos, env.cur_angle, t, dist_err)
        # rendered = plot_lanes(rendered, env, env.cur_pos, env.cur_angle, t, 0, error_frame=rendered)
        own_render(env, rendered)

    if done:
        print("***DONE***")
        env.reset()

x = np.arange(len(errors))

print()
print("stats:")
print()
print("error mean: {}, amount: {}".format(np.mean(errors), len(errors)))
if len(straight_tile_errors) != 0:
    print("straight error mean: {}, amount: {}".format(np.mean(straight_tile_errors), len(straight_tile_errors)))
if len(curve_left_errors) != 0:
    print("curve left error mean: {}, amount: {}".format(np.mean(curve_left_errors), len(curve_left_errors)))
if len(curve_right_errors) != 0:
    print("curve right error mean: {}, amount: {}".format(np.mean(curve_right_errors), len(curve_right_errors)))
if len(three_way_errors) != 0:
    print("three_way error mean: {}, amount: {}".format(np.mean(three_way_errors), len(three_way_errors)))
if len(four_way_errors) != 0:
    print("four_way error mean: {}, amount: {}".format(np.mean(four_way_errors), len(four_way_errors)))


distances, anlges = zip(*errors)

fig, (ax1, ax2) = plt.subplot(1, 2)
plt.title("distances hist")
plt.hist(distances)

plt.figure()
plt.title("scatter")
plt.scatter(distances, all_dist_errors)

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('step')
ax1.set_ylabel('error', color=color)
ax1.plot(x, all_dist_errors, color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax1.axhline(c=color)

ax2 = ax1.twinx()

color = 'tab:blue'
ax2.set_ylabel('distances', color=color)
ax2.plot(x, distances, color=color)
ax2.tick_params(axis='y', labelcolor=color)
ax2.axhline(c=color)

ax3 = ax1.twinx()

color = 'tab:green'
ax3.set_ylabel('angle from straight', color=color)
ax3.plot(x, angles, color=color)
ax3.tick_params(axis='y', labelcolor=color)
ax3.axhline(c=color)

fig.tight_layout()
plt.show()
