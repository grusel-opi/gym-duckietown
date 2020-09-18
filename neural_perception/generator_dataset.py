import tensorflow as tf
from gym_duckietown.envs import DuckietownEnv
import numpy as np
import cv2
import time


AUTOTUNE = tf.data.experimental.AUTOTUNE

TARGET_IMG_WIDTH = 640 // 2
TARGET_IMG_HEIGHT = 480 // 2
IMG_SHAPE = (480, 640, 3)
LABEL_SHAPE = (2,)

K_P = 10
K_D = 1
SPEED = 0.2


class DuckieEnvWrapper:
    def __init__(self):
        self.duckietown = build_duckie_env()
        self.steps = 0
        self.steps_to_reset = 1000

    def generator(self, num_samples=100_000):
        for sample_idx in range(num_samples):
            if self.steps >= self.steps_to_reset:
                self.duckietown.reset()
                self.steps = 0
            self.steps += 1
            lane_pose = self.duckietown.get_lane_pos2(self.duckietown.cur_pos, self.duckietown.cur_angle)
            distance_to_road_center = lane_pose.dist
            angle_from_straight_in_rads = lane_pose.angle_rad
            action = get_action(distance_to_road_center, angle_from_straight_in_rads)
            y = np.array([distance_to_road_center, angle_from_straight_in_rads])
            x, _, _, _ = self.duckietown.step(action)
            yield x, y

    def get_dataset(self):
        return tf.data.Dataset.from_generator(
            self.generator,
            output_types=(tf.dtypes.float32, tf.dtypes.uint8),
            output_shapes=(IMG_SHAPE, LABEL_SHAPE)
        )


def get_action(dist, angle, speed=SPEED, k_p=K_P, k_d=K_D):
    steering = k_p * dist + k_d * angle
    return np.array([speed, steering])


def preprocess(data):
    return np.array([cv2.resize(data, (TARGET_IMG_WIDTH, TARGET_IMG_HEIGHT))])


def build_duckie_env():
    return DuckietownEnv(
        map_name='udem1',
        domain_rand=True,
        draw_bbox=False,
        full_transparency=True
    )


def benchmark(dataset, num_epochs=1):
    start_time = time.perf_counter()
    for epoch_num in range(num_epochs):
        count = 0
        for sample in enumerate(dataset):
            count += 1
            time.sleep(0.01)
    tf.print("Execution time: {}, count: {}".format(time.perf_counter() - start_time, count))


if __name__ == '__main__':
    wrapper = DuckieEnvWrapper()
    ds = wrapper.get_dataset().cache().batch(32).prefetch(buffer_size=AUTOTUNE)
    benchmark(ds)
