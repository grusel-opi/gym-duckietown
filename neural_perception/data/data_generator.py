import sys

from gym_duckietown.simulator import *
from gym_duckietown.graphics import *
from gym_duckietown.envs import DuckietownEnv
from matplotlib import pyplot as plt
from neural_perception.util.util import get_lane_pos
from learning.imitation.iil_dagger.teacher import PurePursuitPolicy


tile_kinds = ['straight', 'curve_left', 'curve_right', '3way', '4way']


class DuckietownImager(DuckietownEnv):
    def __init__(self, path, **kwargs):
        DuckietownEnv.__init__(self, **kwargs)
        self.expert = PurePursuitPolicy(self)
        self.max_steps = math.inf
        self.map_name = 'udem1'
        self.domain_rand = True
        self.draw_bbox = False
        self.full_transparency = True
        self.accept_start_angle_deg = 90
        self.set_size = 1000
        self.path = path
        self.k_p = np.random.uniform(0.8, 1.2)
        self.k_d = np.random.uniform(0.8, 1.2)
        self.action_speed = 0.4
        self.images = np.zeros(shape=(self.set_size, *self.observation_space.shape), dtype=self.observation_space.dtype)
        self.labels = np.zeros(shape=(self.set_size, 3), dtype=np.float32)

    def produce_expert_images(self):
        obs = self.reset()

        for i in range(self.set_size):

            percent = round(i * 100 / self.set_size, 2)
            print(f'\rgenerating set: {percent} %', end='\r')

            action = self.expert.predict()

            self.images[i] = obs
            self.labels[i] = np.array([action[0], action[1], self.get_tile_kind()])

            obs, _, done, _ = self.step(action)

            if done:
                obs = self.reset()

    def produce_rand_expert_images(self, reset_steps=10):
        """
        Do n steps by expert driver then reset to random pose.
        """
        reset_counter = 0
        obs = self.reset()

        for i in range(self.set_size):

            percent = round(i * 100 / self.set_size, 2)
            print(f'\rgenerating set: {percent} %', end='\r')

            action = self.expert.predict()

            self.images[i] = obs
            self.labels[i] = np.array([action[0], action[1], self.get_tile_kind()])

            obs, reward, done, info = self.step(action)

            reset_counter += 1
            if done or reset_counter == reset_steps:
                self.reset()
                reset_counter = 0

    def produce_pd_images(self):
        obs = self.reset()

        for i in range(self.set_size):

            percent = round(i * 100 / self.set_size, 2)
            print(f'\rgenerating set: {percent} %', end='\r')

            lp = get_lane_pos(self)

            while lp.dist_to_edge < 0:
                obs = self.reset()
                lp = get_lane_pos(self)

            self.images[i] = obs
            self.labels[i] = np.array([lp.dist_to_edge * 100, lp.angle_deg, self.get_tile_kind()])

            steering = self.k_p * lp.dist + self.k_d * lp.angle_rad

            obs, reward, done, info = self.step(np.array([self.action_speed, steering]))

            if done:
                self.reset()

    def produce_rand_pd_images(self, reset_steps=10):
        """
        Do n steps by pd controller then reset to random pose.
        """
        reset_counter = 0
        obs = self.reset()

        for i in range(self.set_size):

            percent = round(i * 100 / self.set_size, 2)
            print(f'\rgenerating set: {percent} %', end='\r')

            lp = get_lane_pos(self)

            while lp.dist_to_edge < 0:
                obs = self.reset()
                lp = get_lane_pos(self)

            self.images[i] = obs
            self.labels[i] = np.array([lp.dist_to_edge * 100, lp.angle_deg, self.get_tile_kind()])

            steering = self.k_p * lp.dist + self.k_d * lp.angle_rad

            obs, reward, done, info = self.step(np.array([self.action_speed, steering]))

            reset_counter += 1
            if done or reset_counter == reset_steps:
                self.reset()
                reset_counter = 0

    def generate_and_save(self, num_images, kind):
        if kind == 'pd':
            fn = self.produce_pd_images
        elif kind == 'pd random':
            fn = self.produce_rand_pd_images
        elif kind == 'expert':
            fn = self.produce_expert_images
        elif kind == 'expert random':
            fn = self.produce_rand_expert_images
        else:
            print("Imager kind not recognized..")
            return

        sets = num_images // self.set_size

        try:
            os.mkdir(self.path)
        except OSError as err:
            if os.path.isdir(self.path):
                pass
            else:
                print("OS error: {0}".format(err))
                return

        for set_no in range(sets):
            print(f"Generating set {set_no} of {sets} total (kind: {kind})")
            fn()
            for i in range(self.set_size):
                print(f"\rSaving image no {i + 1 + set_no*self.set_size} of {self.set_size*sets}", end='\r')
                plt.imsave(self.path + str(self.labels[i]) + '.png', self.images[i])

    def get_tile_kind(self):
        kind = self._get_tile(env.cur_pos[0], env.cur_pos[2])['kind']
        if kind.startswith('straight'):
            return 0
        elif kind == 'curve_left':
            return 1
        elif kind == 'curve_right':
            return 2
        elif kind.startswith('3way'):
            return 3
        elif kind.startswith('4way'):
            return 4


if __name__ == '__main__':
    imgs = 100_000
    path = '/home/gandalf/ws/team/datasets/'
    name = 'expert_random/'
    env = DuckietownImager(path=path+name)
    env.generate_and_save(imgs, kind='expert random')
    sys.exit()
