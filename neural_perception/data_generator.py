import sys

from gym_duckietown.simulator import *
from gym_duckietown.graphics import *
from gym_duckietown.envs import DuckietownEnv
from matplotlib import pyplot as plt
from neural_perception.util import get_lane_pos


class ControlledDuckietownImager(DuckietownEnv):
    def __init__(self, **kwargs):
        DuckietownEnv.__init__(self, **kwargs)
        self.max_steps = math.inf
        self.map_name = 'udem1'
        self.domain_rand = True
        self.draw_bbox = False
        self.full_transparency = True
        self.set_size = 100
        self.path = "../../datasets/generated_controlled_dist_to_edge_better_in_cm/"
        self.k_p = 1
        self.k_d = 1
        self.action_speed = 0.2
        self.images = np.zeros(shape=(self.set_size, *self.observation_space.shape),
                               dtype=self.observation_space.dtype)
        self.labels = np.zeros(shape=(self.set_size, 2), dtype=np.float32)

    def produce_images(self, n=10):
        obs = self.reset()
        for i in range(self.set_size):
            for _ in range(n):  # do n steps for every image
                try:
                    lp = get_lane_pos(self)
                except NotInLane:
                    self.reset()
                    continue
                distance_to_road_center = lp.dist
                angle_from_straight_in_rads = lp.angle_rad
                steering = self.k_p * distance_to_road_center + self.k_d * angle_from_straight_in_rads
                action = np.array([self.action_speed, steering])
                obs, reward, done, info = self.step(action)
                if done:
                    print("***DONE***")
                    self.reset()

            if lp.dist_to_edge < 0:
                continue
            else:
                self.images[i] = obs
                self.labels[i] = np.array([lp.dist_to_edge * 100, lp.angle_deg])

    def generate_and_save(self, num_sets):
        try:
            os.mkdir(self.path)
        except OSError:
            if os.path.isdir(self.path):
                pass
            else:
                return
        for _ in range(num_sets):
            self.produce_images()
            for i in range(self.set_size):
                plt.imsave(self.path + str(self.labels[i]) + '.png', self.images[i])


class RandomDuckietownImager(Simulator):

    def __init__(self, **kwargs):
        Simulator.__init__(self, **kwargs)
        self.map_name = 'udem1'
        self.domain_rand = True
        self.draw_bbox = False
        self.full_transparency = True
        self.accept_start_angle_deg = 90
        self.k_p = 1
        self.k_d = 1
        self.action_speed = 0.2
        self.set_size = 1000
        self.path = "../../datasets/random_cm_to_edge_deg_offset205/"
        self.images = np.zeros(shape=(self.set_size, *self.observation_space.shape), dtype=self.observation_space.dtype)
        self.labels = np.zeros(shape=(self.set_size, 2), dtype=np.float32)

    def produce_images(self, reset_counter=0):
        """
        Do n steps by pd controller then reset to random pose.
        """
        obs = self.reset()

        for i in range(self.set_size):

            percent = round(i * 100 / self.set_size, 2)
            print(f'\rgenerating set: {percent} %', end='\r')

            lp = get_lane_pos(self)

            while lp.dist_to_edge < 0:
                self.reset()
                lp = get_lane_pos(self)

            self.images[i] = obs
            self.labels[i] = np.array([lp.dist_to_edge * 100, lp.angle_deg])

            steering = self.k_p * lp.dist + self.k_d * lp.angle_rad

            obs, reward, done, info = self.step(np.array([self.action_speed, steering]))

            reset_counter += 1
            if done or reset_counter >= 10:
                self.reset()

    def generate_and_save(self, num_images):
        sets = num_images // self.set_size
        try:
            os.mkdir(self.path)
        except OSError:
            if os.path.isdir(self.path):
                pass
            else:
                return
        for set_no in range(sets):
            print(f"Generating set {set_no} of {sets} total")
            self.produce_images()
            for i in range(self.set_size):
                print(f"\rSaving image no {i + set_no*self.set_size} of {self.set_size*sets}", end='\r')
                plt.imsave(self.path + str(self.labels[i]) + '.png', self.images[i])


if __name__ == '__main__':
    imgs = 80_000
    env = RandomDuckietownImager()
    env.generate_and_save(imgs)
    sys.exit()
