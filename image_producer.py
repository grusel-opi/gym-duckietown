import argparse
import numpy as np
import gym
import os
from gym_duckietown.envs import DuckietownEnv
from matplotlib import pyplot as PLT


def get_images():

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
            full_transparency=True,
            accept_start_angle_deg=360,
        )
    else:
        env = gym.make(args.env_name)

    images = list()

    for _ in range(num_imgs):
        start = list(env.drivable_tiles[int(np.random.uniform(0, len(env.drivable_tiles)))]['coords'])
        env.user_tile_start = start
        env.reset()
        dist = env.get_agent_info()['Simulator']['lane_position']['dist']
        dot_dir = env.get_agent_info()['Simulator']['lane_position']['dot_dir']
        images.append((env.render(mode='rgb_array'), (dist, dot_dir)))

    return images


if __name__ == '__main__':

    save = True
    num_imgs = 100

    imgs = get_images()

    if save:

        path = "./pictures/"

        try:
            os.mkdir(path)
        except OSError:
            print("./pictures may already be there..")
            exit(-1)

        for img, label in imgs:
            PLT.imsave(path + str(label) + ".png", img)
