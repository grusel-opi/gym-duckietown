import tensorflow as tf
import os

AUTOTUNE = tf.data.experimental.AUTOTUNE

DATA_DIR = "/home/gandalf/ws/team/datasets/expert_tilekind/"

expert_action_size = 79_914

TRAIN_CACHE_FILE = "/home/gandalf/ws/team/gym-duckietown/neural_perception/model/caches/train-expert_action.tfcache"
TEST_CACHE_FILE = "/home/gandalf/ws/team/gym-duckietown/neural_perception/model/caches/test-expert_action.tfcache"
VAL_CACHE_FILE = "/home/gandalf/ws/team/gym-duckietown/neural_perception/model/caches/val-expert_action.tfcache"

ORIG_IMG_SHAPE = (480, 640, 3)
RESIZE_IMG_SHAPE = (120, 160, 3)
TARGET_IMAGE_SHAPE = (80, 160, 3)


@tf.function
def preprocess(image, label):
    height, width, _ = RESIZE_IMG_SHAPE
    image = tf.image.resize(image, (height, width))
    image = image[height // 3:, :]
    return image, label


def get_label(file_path):
    parts = tf.strings.split(file_path, os.path.sep)
    file_name = parts[-1]
    label = tf.strings.regex_replace(input=file_name, pattern='\[|\]|.png', rewrite='')
    label = tf.strings.split(label)
    label = tf.strings.to_number(label)
    return label


def decode_img(img):
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img


def process_path(file_path):
    label = get_label(file_path)
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label


def get_datasets_unprepared(train_fraction=0.7, test_fraction=0.3):
    list_ds = tf.data.Dataset.list_files(DATA_DIR + '*')

    dataset_size = expert_action_size

    full_dataset = list_ds.shuffle(buffer_size=dataset_size)

    train_size = int(train_fraction * dataset_size)
    test_size = int(test_fraction * dataset_size)

    train_dataset = full_dataset.take(train_size)
    test_dataset = full_dataset.skip(train_size)
    val_dataset = test_dataset.skip(test_size)
    test_dataset = test_dataset.take(test_size)

    train_dataset = train_dataset.map(process_path, num_parallel_calls=AUTOTUNE)
    test_dataset = test_dataset.map(process_path, num_parallel_calls=AUTOTUNE)
    val_dataset = val_dataset.map(process_path, num_parallel_calls=AUTOTUNE)

    return train_dataset, test_dataset, val_dataset


def prepare_for_training(ds, batch_size, cache):
    ds = ds.map(preprocess, num_parallel_calls=AUTOTUNE)

    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()

    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds


def get_ds(batch_size):
    train_ds, test_ds, val_ds = get_datasets_unprepared()

    train_ds = prepare_for_training(train_ds, batch_size=batch_size, cache=TRAIN_CACHE_FILE)
    test_ds = prepare_for_training(test_ds, batch_size=batch_size, cache=TEST_CACHE_FILE)
    val_ds = prepare_for_training(val_ds, batch_size=batch_size, cache=VAL_CACHE_FILE)

    return train_ds, test_ds, val_ds
