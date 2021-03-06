#!/usr/bin/env python3

import numpy as np
import os
from gym_duckietown.envs import DuckietownEnv
import tensorflow as tf
import matplotlib.pyplot as plt
import time

from neural_perception.util.util import preprocess, get_lane_pos, own_render, get_mean_and_std

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

env = DuckietownEnv(domain_rand=False,
                    draw_bbox=False)

model = tf.keras.models.load_model(
    '/home/gandalf/ws/team/gym-duckietown/neural_perception/model/saved_model/12.10.2020-15:06:43')

k_p = 1
k_d = 1
speed = 0.2

steps = env.max_steps = 100_000
visual = False
nn_control = True

labels = []
predictions = []

errors = []
straight_tile_errors = []
curve_left_errors = []
curve_right_errors = []
three_way_errors = []
four_way_errors = []

crashes = 0

obs = env.reset()
obs = preprocess(obs)

if visual:
    env.render()

t0 = time.perf_counter()

for i in range(steps):

    percent = round(i * 100 / steps, 2)
    print(f'\rrunning: {percent} %', end='\r')

    lane_pose = get_lane_pos(env)

    distance_to_road_edge = lane_pose.dist_to_edge * 100
    distance_to_road_center = lane_pose.dist
    angle_from_straight_in_rad = lane_pose.angle_rad
    angle_from_straight_in_deg = lane_pose.angle_deg

    y_hat = model(obs)
    d = y_hat[0][0].numpy()
    a = y_hat[0][1].numpy()

    dist_err = abs(distance_to_road_edge - d)
    angle_err = abs(angle_from_straight_in_deg - a)

    errors.append([dist_err, angle_err])
    labels.append([distance_to_road_edge, angle_from_straight_in_deg])
    predictions.append([d, a])

    kind = env._get_tile(env.cur_pos[0], env.cur_pos[2])['kind']

    if kind.startswith('straight'):
        straight_tile_errors.append([dist_err, angle_err])
    elif kind == 'curve_left':
        curve_left_errors.append([dist_err, angle_err])
    elif kind == 'curve_right':
        curve_right_errors.append([dist_err, angle_err])
    elif kind.startswith('3way'):
        three_way_errors.append([dist_err, angle_err])
    elif kind.startswith('4way'):
        four_way_errors.append([dist_err, angle_err])

    if nn_control:
        d_hat_to_center = (d - 20.5) / 100.
        a_hat_in_rad = (a * 2 * np.pi) / 360.
        steering = k_p * d_hat_to_center + k_d * a_hat_in_rad
    else:
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
        print("***CRASHED***")
        crashes += 1
        env.reset()

t1 = time.perf_counter()

x = np.arange(len(errors))
distances, angles = zip(*labels)
distances_err, angles_err = zip(*errors)

print("time: {}".format(t1 - t0))
print()
print("stats:")
print()
print("crashes per steps: {} / {}".format(crashes, steps))
print("error mean: {}, std: {}, min: {}, max: {}, amount: {}".format(np.mean(errors, axis=0), np.std(errors, axis=0),
                                                                     np.min(errors, axis=0), np.max(errors, axis=0),
                                                                     len(errors)))
if len(straight_tile_errors) != 0:
    print("straight error mean: {}, std: {}, min: {}, max: {}, amount: {}".format(np.mean(straight_tile_errors, axis=0),
                                                                                  np.std(straight_tile_errors, axis=0),
                                                                                  np.min(straight_tile_errors, axis=0),
                                                                                  np.max(straight_tile_errors, axis=0),
                                                                                  len(straight_tile_errors)))
if len(curve_left_errors) != 0:
    print("curve left error mean: {}, std: {}, min: {}, max: {}, amount: {}".format(np.mean(curve_left_errors, axis=0),
                                                                                    np.std(curve_left_errors, axis=0),
                                                                                    np.min(curve_left_errors, axis=0),
                                                                                    np.max(curve_left_errors, axis=0),
                                                                                    len(curve_left_errors)))
if len(curve_right_errors) != 0:
    print(
        "curve right error mean: {}, std: {}, min: {}, max: {}, amount: {}".format(np.mean(curve_right_errors, axis=0),
                                                                                   np.std(curve_right_errors, axis=0),
                                                                                   np.min(curve_right_errors, axis=0),
                                                                                   np.max(curve_right_errors, axis=0),
                                                                                   len(curve_right_errors)))
if len(three_way_errors) != 0:
    print("three_way error mean: {}, std: {}, min: {}, max: {}, amount: {}".format(np.mean(three_way_errors, axis=0),
                                                                                   np.std(three_way_errors, axis=0),
                                                                                   np.min(three_way_errors, axis=0),
                                                                                   np.max(three_way_errors, axis=0),
                                                                                   len(three_way_errors)))
if len(four_way_errors) != 0:
    print("four_way error mean: {}, std: {}, min: {}, max: {}, amount: {}".format(np.mean(four_way_errors, axis=0),
                                                                                  np.std(four_way_errors, axis=0),
                                                                                  np.min(four_way_errors, axis=0),
                                                                                  np.max(four_way_errors, axis=0),
                                                                                  len(four_way_errors)))

# histograms for label distribution
fig_hist, (ax_hist_d, ax_hist_a) = plt.subplots(1, 2)

ax_hist_d.set_title("distances distribution")
ax_hist_d.hist(distances)
ax_hist_a.set_title("angles distribution")
ax_hist_a.hist(angles)

# error bar plot for label -> error mapping
fig_scatter, (ax_scat_d, ax_scat_a) = plt.subplots(1, 2)

distances_discrete, distances_err_mean, distances_err_std = get_mean_and_std(distances, distances_err)
angles_discrete, angles_err_mean, angles_err_std = get_mean_and_std(angles, angles_err)

ax_scat_d.set_title("distances error")
ax_scat_d.errorbar(distances_discrete, distances_err_mean, yerr=distances_err_std, fmt='o')
ax_scat_a.set_title("angles error")
ax_scat_a.errorbar(angles_discrete, angles_err_mean, yerr=angles_err_std, fmt='o')

# line plot for (dist, angle) -> dist err
# and (dist, angle) -> angle err visualization
fig_line, (ax_line_d_0, ax_line_a_0) = plt.subplots(2, 1)

color = 'tab:red'
ax_line_d_0.set_xlabel('step')
ax_line_d_0.set_ylabel('d error', color=color)
ax_line_d_0.plot(x, distances_err, color=color)
ax_line_d_0.tick_params(axis='y', labelcolor=color)
ax_line_d_0.axhline(c=color)

ax_line_d_1 = ax_line_d_0.twinx()
color = 'tab:blue'
ax_line_d_1.set_ylabel('distances', color=color)
ax_line_d_1.plot(x, distances, color=color)
ax_line_d_1.tick_params(axis='y', labelcolor=color)
ax_line_d_1.axhline(y=25, c=color)

ax_line_d_2 = ax_line_d_1.twinx()
color = 'tab:green'
ax_line_d_2.set_ylabel('angle from straight', color=color)
ax_line_d_2.plot(x, angles, color=color)
ax_line_d_2.tick_params(axis='y', labelcolor=color)
ax_line_d_2.axhline(c=color)

color = 'tab:red'
ax_line_a_0.set_xlabel('step')
ax_line_a_0.set_ylabel('a error', color=color)
ax_line_a_0.plot(x, angles_err, color=color)
ax_line_a_0.tick_params(axis='y', labelcolor=color)
ax_line_a_0.axhline(c=color)

ax_line_a_1 = ax_line_a_0.twinx()
color = 'tab:blue'
ax_line_a_1.set_ylabel('distances', color=color)
ax_line_a_1.plot(x, distances, color=color)
ax_line_a_1.tick_params(axis='y', labelcolor=color)
ax_line_a_1.axhline(y=25, c=color)

ax_line_a_2 = ax_line_a_1.twinx()

color = 'tab:green'
ax_line_a_2.set_ylabel('angle from straight', color=color)
ax_line_a_2.plot(x, angles, color=color)
ax_line_a_2.tick_params(axis='y', labelcolor=color)
ax_line_a_2.axhline(c=color)

fig_line.tight_layout()

# fig3d = plt.figure()
#
# ax3d = fig3d.add_subplot(111, projection='3d')
#
# xs3d = list(distances)
# ys3d = list(angles)
# zs3d_d, zs3d_a = zip(*errors)
#
# ax3d.scatter(xs3d, ys3d, list(zs3d_d), marker='o')
# ax3d.scatter(xs3d, ys3d, list(zs3d_a), marker='^')
#
# ax3d.set_xlabel('X Label')
# ax3d.set_ylabel('Y Label')
# ax3d.set_zlabel('Z Label')

plt.show()
