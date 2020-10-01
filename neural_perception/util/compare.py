import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from gym_duckietown.envs import DuckietownEnv
from neural_perception.control_test import preprocess as integration_preprocess
from neural_perception.control_test import get_lane_pos
from neural_perception.control_test import RESIZE_IMG_SHAPE

from data import get_ds, get_datasets_unprepared
from data import preprocess as training_preprocess

from welford import Welford


def get_training_dataset(num_samples):
    _, test_ds, _ = get_datasets_unprepared()
    ds = test_ds.take(num_samples)
    # ds = ds.map(training_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.batch(1)
    return ds


def get_integration_dataset(num_samples):

    period = 10
    data = np.zeros(shape=(num_samples, 1, 480, 640, 3))
    labels = np.zeros(shape=(num_samples, 1))

    k_p = 1
    k_d = 1
    speed = 0.2

    env = DuckietownEnv(domain_rand=False, draw_bbox=False)
    iterations = env.max_steps = num_samples * period
    env.reset()

    for i in range(iterations):

        percent = round(i * 100 / iterations, 2)
        print(f'\rsimulator running: {percent} %', end='\r')

        lane_pose = get_lane_pos(env)
        distance_to_road_center = lane_pose.dist
        angle_from_straight_in_rad = lane_pose.angle_rad

        steering = k_p * distance_to_road_center + k_d * angle_from_straight_in_rad
        command = np.array([speed, steering])
        obs, _, done, _ = env.step(command)
        # obs = integration_preprocess(obs)

        if i % period == 0:
            data[i // period, 0, :, :, :] = obs
            labels[i // period][0] = lane_pose.dist_to_edge*100

        if done:
            env.reset()

    return list(zip(data, labels))


def get_model_error(model, ds, iterations):
    errors = []
    for i, (element, label) in enumerate(ds):
        percent = round(i * 100 / iterations, 2)
        print(f'\rintegration test running: {percent} %', end='\r')
        y_hat = model(element, training=False)
        errors.append(abs(label - y_hat))

    print('\r\n')
    return errors


def get_welford(ds):
    features = [f for f, l in ds]
    welford = Welford(shape=(480, 640, 3), lst=features)
    return welford.mean, welford.std


if __name__ == '__main__':

    num_samples = 1000

    model = tf.keras.models.load_model('/home/gandalf/ws/team/lean-test/saved_model/14.09.2020-14:57:57/')

    integration_dataset = get_integration_dataset(num_samples)
    training_dataset = get_training_dataset(num_samples)

    m1, s1 = get_welford(integration_dataset)
    m2, s2 = get_welford(training_dataset)

    print("integration mean mean: {}, std mean: {}".format(np.mean(m1), np.mean(s1)))
    print("training mean mean: {}, std mean: {}".format(np.mean(m2), np.mean(s2)))
    print()
    print("integration mean min: {}, max: {}".format(np.min(m1), np.max(m1)))
    print("training mean min: {}, max: {}".format(np.min(m2), np.max(m2)))
    print("integration std min: {}, max: {}".format(np.min(s1), np.max(s1)))
    print("training std min: {}, max: {}".format(np.min(s2), np.max(s2)))

    plt.matshow(m1[0], fignum=0)
    plt.matshow(s1[0], fignum=1)
    plt.matshow(m2[0], fignum=2)
    plt.matshow(s2[0], fignum=3)

    # m1_pic, s1_pic = m1 * 255., s1 * 255.
    # m2_pic, s2_pic = m2 * 255., s2 * 255.
    #
    # plt.matshow(m1_pic[0], fignum=0)
    # plt.matshow(s1_pic[0], fignum=1)
    # plt.matshow(m2_pic[0], fignum=2)
    # plt.matshow(s2_pic[0], fignum=3)

    plt.show()

    # integration_errors = get_model_error(model, integration_dataset, num_samples)
    # training_errors = get_model_error(model, training_dataset, num_samples)
    #
    # print("integration error: ", np.mean(integration_errors))
    #
    # print("training error: ", np.mean(training_errors))
