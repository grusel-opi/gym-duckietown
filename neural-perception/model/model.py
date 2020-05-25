import tensorflow as tf

# TODO: we should handle imports in a better way, so that we can call the model from other files without chaning the
#         import! (import data.data_generator fix the issue but then we cannot run the main file here)


class PoseRegress(tf.keras.Model):
    def __init__(self, config):
        super(PoseRegress, self).__init__()

        self.pool1 = tf.keras.layers.MaxPool2D(2)  # halve input dimensions

        self.conv1 = tf.keras.layers.Conv2D(16, kernel_size=3, padding='same', activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(32, kernel_size=3, strides=2, padding='same', activation='relu')
        self.conv3 = tf.keras.layers.Conv2D(64, kernel_size=11, padding='same', activation='relu')
        self.conv4 = tf.keras.layers.Conv2D(64, kernel_size=11, strides=2, padding='same', activation='relu')
        self.conv5 = tf.keras.layers.Conv2D(128, kernel_size=3, padding='same', activation='relu')
        self.conv6 = tf.keras.layers.Conv2D(128, kernel_size=1, strides=2, padding='same', activation='relu')
        self.conv7 = tf.keras.layers.Conv2D(256, kernel_size=1, padding='same', activation='relu')

        # encoder: fully convolutional
        # self.conv1 = tf.keras.layers.Conv2D(64, kernel_size=3, padding='same', activation='relu')
        # self.conv2 = tf.keras.layers.Conv2D(128, kernel_size=3, strides=2, padding='same', activation='relu')
        # self.conv3 = tf.keras.layers.Conv2D(128, kernel_size=3, strides=2, padding='same', activation='relu')
        # self.conv4 = tf.keras.layers.Conv2D(256, kernel_size=3, padding='same', activation='relu')
        # self.conv5 = tf.keras.layers.Conv2D(256, kernel_size=3, strides=2, padding='same', activation='relu')
        # self.conv6 = tf.keras.layers.Conv2D(512, kernel_size=3, padding='same', activation='relu')
        # self.conv7 = tf.keras.layers.Conv2D(512, kernel_size=3, padding='same', activation='relu')
        # self.conv8 = tf.keras.layers.Conv2D(512, kernel_size=3, padding='same', activation='relu')
        # self.conv8 = tf.keras.layers.Conv2D(512, kernel_size=3, padding='same', activation='relu')
        # self.conv9 = tf.keras.layers.Conv2D(4096, kernel_size=1, activation='relu')

        self.flatter = tf.keras.layers.Flatten()

        self.localizer = tf.keras.layers.Dense(1024)

        self.regressor_d_1 = tf.keras.layers.Dense(512)
        self.regressor_a_1 = tf.keras.layers.Dense(512)

        self.regressor_d_2 = tf.keras.layers.Dense(256)
        self.regressor_a_2 = tf.keras.layers.Dense(256)

        self.out_d = tf.keras.layers.Dense(1)
        self.out_a = tf.keras.layers.Dense(1)

    def call(self, inputs, **kwargs):

        x = self.pool1(inputs)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)

        x = self.flatter(x)

        x = self.localizer(x)

        d = self.regressor_d_1(x)
        a = self.regressor_a_1(x)

        d = self.regressor_d_2(d)
        a = self.regressor_a_2(a)

        d = self.out_d(d)
        a = self.out_a(a)

        return tf.keras.layers.concatenate([d.output, a.utput])

