#!/usr/bin/env python3

import numpy as np
import os
from gym_duckietown.envs import DuckietownEnv
import math

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def angle_diff(a, b):
    c = b - a
    return (c + math.pi) % math.pi * 2 - math.pi


def get_action(env, step_factor=0.1, v_max=1, omega_max=math.pi, k_v=1, k_omega=1):
    pos, angle = env.cur_pos, env.cur_angle
    closest_point, tangent = env.closest_curve_point(pos, angle)
    step = env.road_tile_size * step_factor
    target_point = closest_point + step
    target_point, _ = env.closest_curve_point(target_point, angle)
    x, _, y = pos
    x_hat, _, y_hat = target_point
    v = max(min(k_v * math.sqrt((x - x_hat) ** 2 + (y - y_hat) ** 2), v_max), -v_max)
    angle_hat = math.atan2(y_hat - y, x_hat - x)
    omega = max(min(k_omega * angle_diff(angle_hat, angle), omega_max), -omega_max)
    return np.array([v, omega])


if __name__ == '__main__':
    env = DuckietownEnv(domain_rand=False,
                        draw_bbox=False)
    obs = env.reset()
    env.render()

    steps = env.max_steps = 10_000

    for i in range(steps):

        action = get_action(env)

        action[0] = 0.2

        obs, _, done, _ = env.step(action)

        env.render()

        if done:
            env.reset()
