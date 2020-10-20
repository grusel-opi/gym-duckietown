#!/usr/bin/env python3

import numpy as np
import os
from gym_duckietown.envs import DuckietownEnv
import tensorflow as tf
import matplotlib.pyplot as plt
import time

from learning.imitation.iil_dagger.teacher import PurePursuitPolicy
from neural_perception.util.util import preprocess, get_lane_pos, own_render, get_mean_and_std

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

env = DuckietownEnv(domain_rand=False,
                    draw_bbox=False)

model = tf.keras.models.load_model(
    '/home/gandalf/ws/team/gym-duckietown/neural_perception/model/saved_model/14.10.2020-18:53:08')

expert = PurePursuitPolicy(env)

speed = 0.2

steps = env.max_steps = 1_000
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

    omega = expert.predict()

    omega_hat = model(obs)[0][0].numpy()

    omega_err = abs(omega_hat - omega)

    errors.append(omega_err)
    labels.append(omega)
    predictions.append(omega_hat)

    kind = env._get_tile(env.cur_pos[0], env.cur_pos[2])['kind']

    if kind.startswith('straight'):
        straight_tile_errors.append(omega_err)
    elif kind == 'curve_left':
        curve_left_errors.append(omega_err)
    elif kind == 'curve_right':
        curve_right_errors.append(omega_err)
    elif kind.startswith('3way'):
        three_way_errors.append(omega_err)
    elif kind.startswith('4way'):
        four_way_errors.append(omega_err)

    if nn_control:
        correction = omega_hat
    else:
        correction = omega

    action = np.array([speed, correction])

    obs, _, done, _ = env.step(action)

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
fig_hist, ax_hist_d = plt.subplots(1, 1)

ax_hist_d.set_title("ideal angular velocities distribution")
ax_hist_d.hist(labels)

# error bar plot for label -> error mapping
fig_scatter, ax_scat_d = plt.subplots(1, 1)

distances_discrete, distances_err_mean, distances_err_std = get_mean_and_std(labels, errors)

ax_scat_d.set_title("omega error")
ax_scat_d.errorbar(distances_discrete, distances_err_mean, yerr=distances_err_std, fmt='o')

# line plot for (dist, angle) -> dist err
# and (dist, angle) -> angle err visualization
fig_line, ax_line_d_0 = plt.subplots(1, 1)

color = 'tab:red'
ax_line_d_0.set_xlabel('step')
ax_line_d_0.set_ylabel('omega error', color=color)
ax_line_d_0.plot(x, errors, color=color)
ax_line_d_0.tick_params(axis='y', labelcolor=color)
ax_line_d_0.axhline(c=color)

ax_line_d_1 = ax_line_d_0.twinx()
color = 'tab:blue'
ax_line_d_1.set_ylabel('angular velocities', color=color)
ax_line_d_1.plot(x, labels, color=color)
ax_line_d_1.tick_params(axis='y', labelcolor=color)
ax_line_d_1.axhline(y=25, c=color)

fig_line.tight_layout()

plt.show()
