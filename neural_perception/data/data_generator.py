import sys

from gym_duckietown.simulator import *
from gym_duckietown.graphics import *
from gym_duckietown.envs import DuckietownEnv
from matplotlib import pyplot as plt
from neural_perception.util.util import get_lane_pos
from learning.imitation.iil_dagger.teacher import PurePursuitPolicy


class ControlledDuckietownImager(DuckietownEnv):
    def __init__(self, **kwargs):
        DuckietownEnv.__init__(self, **kwargs)
        self.max_steps = math.inf
        self.map_name = 'udem1'
        self.domain_rand = True
        self.draw_bbox = False
        self.full_transparency = True
        self.set_size = 1000
        self.path = "../../datasets/pid_off205/"
        self.k_p = 1
        self.k_d = 1
        self.action_speed = 0.2
        self.images = np.zeros(shape=(self.set_size, *self.observation_space.shape),
                               dtype=self.observation_space.dtype)
        self.labels = np.zeros(shape=(self.set_size, 2), dtype=np.float32)

    def produce_images(self, n=10):
        obs = self.reset()

        for i in range(self.set_size):
            lp = None

            for _ in range(n):  # do n steps between every image

                lp = get_lane_pos(self)

                distance_to_road_center = lp.dist
                angle_from_straight_in_rads = lp.angle_rad

                steering = self.k_p * distance_to_road_center + self.k_d * angle_from_straight_in_rads

                obs, reward, done, info = self.step(np.array([self.action_speed, steering]))

                if done:
                    self.reset()

            while lp.dist_to_edge < 0:
                self.reset()
                lp = get_lane_pos(self)

            self.images[i] = obs
            self.labels[i] = np.array([lp.dist_to_edge * 100, lp.angle_deg])


class RandomDuckietownImager(Simulator):

    def __init__(self, **kwargs):
        Simulator.__init__(self, **kwargs)
        self.max_steps = math.inf
        self.map_name = 'udem1'
        self.domain_rand = True
        self.draw_bbox = False
        self.full_transparency = True
        self.accept_start_angle_deg = 90
        self.k_p = 1
        self.k_d = 1
        self.action_speed = 0.2
        self.set_size = 1000
        self.path = "../../../datasets/random_cm_to_edge_deg_offset205/"
        self.images = np.zeros(shape=(self.set_size, *self.observation_space.shape), dtype=self.observation_space.dtype)
        self.labels = np.zeros(shape=(self.set_size, 2), dtype=np.float32)

    def produce_images(self, reset_steps=10):
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
                self.reset()
                lp = get_lane_pos(self)

            self.images[i] = obs
            self.labels[i] = np.array([lp.dist_to_edge * 100, lp.angle_deg])

            steering = self.k_p * lp.dist + self.k_d * lp.angle_rad

            obs, reward, done, info = self.step(np.array([self.action_speed, steering]))

            reset_counter += 1
            if done or reset_counter == reset_steps:
                self.reset()


class ExpertDuckietownImager(Simulator):

    def __init__(self, **kwargs):
        Simulator.__init__(self, **kwargs)
        self.expert = PurePursuitPolicy(self)
        self.max_steps = math.inf
        self.map_name = 'udem1'
        self.domain_rand = True
        self.draw_bbox = False
        self.full_transparency = True
        self.accept_start_angle_deg = 90
        self.set_size = 1000
        self.path = "/home/gandalf/ws/team/datasets/expert_action/"
        self.images = np.zeros(shape=(self.set_size, *self.observation_space.shape), dtype=self.observation_space.dtype)
        self.labels = np.zeros(shape=(self.set_size, 2), dtype=np.float32)

    def produce_images(self):
        obs = self.reset()

        for i in range(self.set_size):

            percent = round(i * 100 / self.set_size, 2)
            print(f'\rgenerating set: {percent} %', end='\r')

            action = self.expert.predict()

            self.images[i] = obs
            self.labels[i] = action

            obs, _, done, _ = self.step(action)

            if done:
                obs = self.reset()


def generate_and_save(imager, num_images):
    sets = num_images // imager.set_size
    try:
        os.mkdir(imager.path)
    except OSError as err:
        if os.path.isdir(imager.path):
            pass
        else:
            print("OS error: {0}".format(err))
            return
    for set_no in range(sets):
        print(f"Generating set {set_no} of {sets} total")
        imager.produce_images()
        for i in range(imager.set_size):
            print(f"\rSaving image no {i + 1 + set_no*imager.set_size} of {imager.set_size*sets}", end='\r')
            plt.imsave(imager.path + str(imager.labels[i]) + '.png', imager.images[i])


if __name__ == '__main__':
    imgs = 80_000
    env = ExpertDuckietownImager()
    generate_and_save(env, imgs)
    sys.exit()
