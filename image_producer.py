import numpy as np
import os
from gym_duckietown.envs import DuckietownEnv
from matplotlib import pyplot as plt


def get_images():
    env = DuckietownEnv(
        map_name='udem1',
        domain_rand=True,
        draw_bbox=False,
        full_transparency=True,
        accept_start_angle_deg=360,
    )

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
            plt.imsave(path + str(label) + ".png", img)
