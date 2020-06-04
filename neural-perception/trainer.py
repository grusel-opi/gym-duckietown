import datetime
import tensorflow as tf
from .data_loader import get_ds
from .model import PoseRegress
from tensorflow.keras.utils import Progbar

current_time = datetime.datetime.now().strftime("%d.%m.%Y-%H:%M:%S")
model_path = "saved_model/" + current_time + "/"
log_dir = "logs/" + current_time + "/"

learning_rate = 0.0001
save = True

EPOCHS = 20
checkpoint_after_n_epochs = 5

train_ds, test_ds, val_ds = get_ds()


# L2 loss, but scaled x 100
@tf.function
def loss_l2(y_true, y_pred):
    return tf.reduce_mean(tf.square((tf.multiply(y_true, 100)) - (tf.multiply(y_pred, 100))))


model = PoseRegress()

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
loss_fn = loss_l2

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_abs_error = tf.keras.metrics.MeanAbsoluteError(name='train_abs_error')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_abs_error = tf.keras.metrics.MeanAbsoluteError(name='test_abs_error')


@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        predictions = model(x, training=True)
        loss = loss_fn(y, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_abs_error(y, predictions)


@tf.function
def test_step(x, y):
    predictions = model(x)
    t_loss = loss_fn(y, predictions)

    test_loss(t_loss)
    test_abs_error(y, predictions)


train_summary_writer = tf.summary.create_file_writer(log_dir + "train/")
test_summary_writer = tf.summary.create_file_writer(log_dir + "test/")

for epoch in range(EPOCHS):

    print("Epoch: ", epoch + 1)

    start = tf.timestamp()

    # Reset the metrics at the start of the next epoch
    train_loss.reset_states()
    train_abs_error.reset_states()
    test_loss.reset_states()
    test_abs_error.reset_states()

    # Set up the progress bars
    train_progress_bar = Progbar(len(list(train_ds)), stateful_metrics=['Loss', 'Mean Abs. Error'])
    test_progress_bar = Progbar(len(list(test_ds)), stateful_metrics=['Loss', 'Mean Abs. Error'])

    step = 0
    for images, labels in train_ds:
        train_step(images, labels)
        values = [('Loss', train_loss.result()), ('Mean Abs. Error', train_abs_error.result() * 100)]
        train_progress_bar.add(step + 1, values=values)
    with train_summary_writer.as_default():
        tf.summary.scalar('Loss', train_loss.result(), step=epoch)
        tf.summary.scalar('Mean Abs. Error', train_abs_error.result() * 100, step=epoch)

    step = 0
    for test_images, test_labels in test_ds:
        test_step(test_images, test_labels)
        values = [('Loss', test_loss.result()), ('Mean Abs. Error', test_abs_error.result() * 100)]
        test_progress_bar.add(step + 1, values=values)
    with test_summary_writer.as_default():
        tf.summary.scalar('Loss', test_loss.result(), step=epoch)
        tf.summary.scalar('Mean Abs. Error', test_abs_error.result() * 100, step=epoch)

    end = tf.timestamp()

    print("Time: {:.2f} [s]".format(end - start))

    if save and ((epoch + 1) % checkpoint_after_n_epochs == 0 or (epoch + 1) == EPOCHS):
        print("Saving model ....")
        model.save(model_path)
