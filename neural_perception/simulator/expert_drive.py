#!/usr/bin/env python3

import numpy as np
import os
from gym_duckietown.envs import DuckietownEnv
from gym_duckietown.graphics import bezier_draw
import math

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def angle_diff(a, b):
    c = a - b
    return (c + math.pi) % (math.pi * 2) - math.pi


def get_action(env, step_factor=0.1, v_max=1, omega_max=math.pi, k_v=1, k_omega=1):
    pos, angle = env.cur_pos, env.cur_angle
    closest_point, tangent = env.closest_curve_point(pos, angle)
    step_size = env.road_tile_size * step_factor
    scaled_tangent = step_size * tangent
    next_point = closest_point + scaled_tangent
    target_point, _ = env.closest_curve_point(next_point, angle)
    x, _, y = pos
    x_hat, _, y_hat = target_point
    x_translate = x_hat - x
    y_translate = y_hat - y
    v = max(min(k_v * math.sqrt(x_translate ** 2 + y_translate ** 2), v_max), -v_max)
    angle_hat = math.atan2(y_translate, x_translate)
    omega = max(min(k_omega * angle_diff(angle_hat, angle), omega_max), -omega_max)
    return np.array([v, omega]), target_point


def lerp(v0, v1, t):
    return (1 - t) * v0 + t * v1


if __name__ == '__main__':
    env = DuckietownEnv(domain_rand=False,
                        draw_bbox=False)
    obs = env.reset()
    env.render()

    steps = env.max_steps = 10_000

    for i in range(steps):

        action, target = get_action(env)

        action[0] = 0.1

        cps = np.array([lerp(env.cur_pos, target, t) for t in np.linspace(0, 1, 4)])

        obs, _, done, _ = env.step(action)

        env.render(own_cps=cps)

        if done:
            env.reset()
