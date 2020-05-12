import argparse
import numpy as np
import gym
from gym_duckietown.envs import DuckietownEnv
from matplotlib import pyplot as PLT

parser = argparse.ArgumentParser()
parser.add_argument('--env-name', default=None)
parser.add_argument('--map-name', default='udem1')
parser.add_argument('--no-pause', action='store_true', help="don't pause on failure")
args = parser.parse_args()

if args.env_name is None:
    env = DuckietownEnv(
        map_name=args.map_name,
        domain_rand=False,
        draw_bbox=False,
        full_transparency=True
    )
else:
    env = gym.make(args.env_name)

num_imgs = 1
imgs = list()

for i in range(num_imgs):
    start = list(env.drivable_tiles[int(np.random.uniform(0, len(env.drivable_tiles)))]['coords'])
    env.user_tile_start = start
    env.reset()
    lane_pos = env.get_agent_info()['Simulator']['lane_position']
    label = (lane_pos.dist, lane_pos.dot_dir)
    imgs.append((env.render(mode='rgb_array'), label))


