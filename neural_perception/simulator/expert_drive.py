#!/usr/bin/env python3

import numpy as np
import os
from gym_duckietown.envs import DuckietownEnv
from neural_perception.util.util import lerp
from learning.imitation.iil_dagger.teacher.pure_pursuit_policy import PurePursuitPolicy
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

if __name__ == '__main__':
    env = DuckietownEnv(domain_rand=False,
                        draw_bbox=False)

    expert = PurePursuitPolicy(env=env)

    obs = env.reset()
    env.render()

    steps = env.max_steps = 10_000

    for i in range(steps):

        action = expert.predict()

        obs, _, done, _ = env.step(action)

        env.render()

        if done:
            env.reset()
