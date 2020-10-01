import datetime
import tensorflow as tf
from neural_perception.model.data import get_ds, TARGET_IMAGE_SHAPE

from tensorflow.keras.layers import Conv2D, Dense, Flatten, Dropout, LayerNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras import Model
from tensorflow.keras.utils import Progbar


class PoseRegress(Model):
    def __init__(self):
        super(PoseRegress, self).__init__()

        self.norm = LayerNormalization(input_shape=TARGET_IMAGE_SHAPE)
        self.conv1 = Conv2D(24, (5, 5), strides=(2, 2), padding='same', kernel_regularizer=l2(0.001), activation='elu')
        self.conv2 = Conv2D(36, (5, 5), strides=(2, 2), padding='same', kernel_regularizer=l2(0.001),  activation='elu')
        self.conv3 = Conv2D(48, (5, 5), strides=(2, 2), padding='same', kernel_regularizer=l2(0.001),  activation='elu')
        self.conv4 = Conv2D(64, (3, 3),  kernel_regularizer=l2(0.001), activation='elu')
        self.drop1 = Dropout(0.2)
        self.conv5 = Conv2D(64, (3, 3),  kernel_regularizer=l2(0.001), activation='elu')

        self.flatter = Flatten()
        self.drop2 = Dropout(0.2)

        self.reg_1 = Dense(100, kernel_regularizer=l2(0.001), activation='elu', )
        self.reg_2 = Dense(50,  kernel_regularizer=l2(0.001), activation='elu')
        self.reg_3 = Dense(10,  kernel_regularizer=l2(0.001), activation='elu')

        self.out = Dense(2)

    def call(self, x, training=False, **kwargs):

        x = self.norm(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        if training:
            x = self.drop1(x)

        x = self.conv5(x)

        x = self.flatter(x)

        if training:
            x = self.drop2(x)

        x = self.reg_1(x)
        x = self.reg_2(x)
        x = self.reg_3(x)

        return self.out(x)


@tf.function
def train_step(x, y):

    with tf.GradientTape() as tape:
        y_hat = model(x, training=True)
        loss = loss_fn(y, y_hat)

    gradients_d = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients_d, model.trainable_variables))

    train_loss(loss)

    d = y[:, 0]
    a = y[:, 1]

    d_hat = y_hat[:, 0]
    a_hat = y_hat[:, 1]

    train_abs_error_d(d, d_hat)
    train_abs_error_a(a, a_hat)


@tf.function
def test_step(x, y):

    y_hat = model(x)

    loss = loss_fn(y, y_hat)
    test_loss(loss)

    d = y[:, 0]
    a = y[:, 1]

    d_hat = y_hat[:, 0]
    a_hat = y_hat[:, 1]

    test_abs_error_d(d, d_hat)
    test_abs_error_a(a, a_hat)


if __name__ == '__main__':

    current_time = datetime.datetime.now().strftime("%d.%m.%Y-%H:%M:%S")
    model_path = "saved_model/" + current_time + "/"
    log_dir = "logs-single-loss/" + current_time + "-layernorm-bs32-lr0002-elu-MSE-concatDS-l2/"

    batch_size = 32
    learning_rate = 0.0002
    EPOCHS = 50
    save = True

    train_ds, test_ds, val_ds = get_ds(batch_size)

    model = PoseRegress()

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    loss_fn = tf.keras.losses.MeanSquaredError()

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    test_loss = tf.keras.metrics.Mean(name='test_loss')

    train_abs_error_d = tf.keras.metrics.MeanAbsoluteError(name='train_abs_error')
    train_abs_error_a = tf.keras.metrics.MeanAbsoluteError(name='train_abs_error')

    test_abs_error_d = tf.keras.metrics.MeanAbsoluteError(name='test_abs_error')
    test_abs_error_a = tf.keras.metrics.MeanAbsoluteError(name='test_abs_error')

    train_summary_writer = tf.summary.create_file_writer(log_dir + "train/")
    test_summary_writer = tf.summary.create_file_writer(log_dir + "test/")

    for epoch in range(EPOCHS):

        print("Epoch: ", epoch + 1)

        start = tf.timestamp()

        train_loss.reset_states()
        test_loss.reset_states()

        train_abs_error_d.reset_states()
        test_abs_error_d.reset_states()

        train_abs_error_a.reset_states()
        test_abs_error_a.reset_states()

        train_progress_bar = Progbar(len(list(train_ds)), stateful_metrics=['Loss', 'Mean Abs. Error'])
        test_progress_bar = Progbar(len(list(test_ds)), stateful_metrics=['Loss', 'Mean Abs. Error'])

        step = 0
        for images, labels in train_ds:
            train_step(images, labels)
            values = [('Loss', train_loss.result()),
                      ('Mean Abs. Error d', train_abs_error_d.result()),
                      ('Mean Abs. Error a', train_abs_error_a.result())]
            train_progress_bar.add(step + 1, values=values)

        with train_summary_writer.as_default():
            tf.summary.scalar('Loss', train_loss.result(), step=epoch)
            tf.summary.scalar('Mean Abs. Error d', train_abs_error_d.result(), step=epoch)
            tf.summary.scalar('Mean Abs. Error a', train_abs_error_a.result(), step=epoch)

        step = 0
        for test_images, test_labels in test_ds:
            test_step(test_images, test_labels)
            values = [('Loss', test_loss.result()),
                      ('Mean Abs. Error d', test_abs_error_d.result()),
                      ('Mean Abs. Error a', test_abs_error_a.result())]
            test_progress_bar.add(step + 1, values=values)

        with test_summary_writer.as_default():
            tf.summary.scalar('Loss', test_loss.result(), step=epoch)
            tf.summary.scalar('Mean Abs. Error d', test_abs_error_d.result(), step=epoch)
            tf.summary.scalar('Mean Abs. Error a', test_abs_error_a.result(), step=epoch)

        end = tf.timestamp()

        print("Time: {:.2f} [s]".format(end - start))

    if save:
        print("Saving model ....")
        model.save(model_path)
