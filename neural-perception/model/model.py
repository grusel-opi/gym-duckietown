import tensorflow as tf
import sys
import numpy as np

sys.path.append('../data/')

import data_generator


class PoseRegress(tf.keras.Model):
    def __init__(self):
        super(PoseRegress, self).__init__()

        # encoder: fully convolutional
        self.conv1 = tf.keras.layers.Conv2D(64, kernel_size=3, padding='same', activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(128, kernel_size=3, strides=2, padding='same', activation='relu')

        self.conv3 = tf.keras.layers.Conv2D(128, kernel_size=3, strides=2, padding='same', activation='relu')

        self.conv4 = tf.keras.layers.Conv2D(256, kernel_size=3, padding='same', activation='relu')
        self.conv5 = tf.keras.layers.Conv2D(256, kernel_size=3, strides=2, padding='same', activation='relu')

        self.conv6 = tf.keras.layers.Conv2D(512, kernel_size=3, padding='same', activation='relu')
        self.conv7 = tf.keras.layers.Conv2D(512, kernel_size=3, padding='same', activation='relu')
        self.conv8 = tf.keras.layers.Conv2D(512, kernel_size=3, padding='same', activation='relu')

        self.conv8 = tf.keras.layers.Conv2D(512, kernel_size=3, padding='same', activation='relu')

        self.conv9 = tf.keras.layers.Conv2D(4096, kernel_size=1, activation='relu')

        self.flatter = tf.keras.layers.Flatten()

        self.localizer = tf.keras.layers.Dense(512)

        self.regressor_d = tf.keras.layers.Dense(512)
        self.regressor_a = tf.keras.layers.Dense(512)

        self.out_d = tf.keras.layers.Dense(1)
        self.out_a = tf.keras.layers.Dense(1)

    def call(self, inputs, **kwargs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        # x = self.conv6(x)
        # x = self.conv7(x)
        # x = self.conv8(x)

        x = self.flatter(x)
        x = self.localizer(x)

        d = self.regressor_d(x)
        a = self.regressor_d(x)

        d = self.out_d(d)
        a = self.out_a(a)
        return d, a


if __name__ == '__main__':
    data, _ = data_generator.get_in_ram_sample(10)
    data = data / 255.
    data = data[:1]
    model = PoseRegress()
    out_d, out_a = model.call(data[:2])
    print(str(out_d.shape))
    print(str(out_a.shape))
    print(out_d)
    print(out_a)

