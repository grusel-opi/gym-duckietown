import tensorflow as tf
import os
import sys

AUTOTUNE = tf.data.experimental.AUTOTUNE

MAX_SET_SIZE = 60_000
train_size = int(0.7 * MAX_SET_SIZE)
val_size = int(0 * MAX_SET_SIZE)
test_size = int(0.3 * MAX_SET_SIZE)

BATCH_SIZE = 32
DATA_DIR = "../generated/"
TRAIN_CACHE_FILE = "./duckie-train.tfcache"
TEST_CACHE_FILE = "./duckie-test.tfcache"
VAL_CACHE_FILE = "./duckie-val.tfcache"
TARGET_IMG_WIDTH = 640 // 2
TARGET_IMG_HEIGHT = 480 // 2


def get_label(file_path):
    parts = tf.strings.split(file_path, os.path.sep)
    file_name = parts[-1]
    label = tf.strings.regex_replace(input=file_name, pattern='\[|\]|.png', rewrite='')
    label = tf.strings.split(label)
    label = tf.strings.to_number(label)
    return label


def decode_img(img):
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return tf.image.resize(img, [TARGET_IMG_WIDTH, TARGET_IMG_HEIGHT])


def process_path(file_path):
    label = get_label(file_path)
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label


def get_datasets_unprepared(size=MAX_SET_SIZE, shuffle_buffer_size=MAX_SET_SIZE // 2):
    list_ds = tf.data.Dataset.list_files(DATA_DIR + '*').take(size)
    full_dataset = list_ds.shuffle(buffer_size=shuffle_buffer_size)

    train_dataset = full_dataset.take(train_size)
    test_dataset = full_dataset.skip(train_size)
    val_dataset = test_dataset.skip(test_size)
    test_dataset = test_dataset.take(test_size)

    train_dataset = train_dataset.map(process_path, num_parallel_calls=AUTOTUNE)
    test_dataset = test_dataset.map(process_path, num_parallel_calls=AUTOTUNE)
    val_dataset = val_dataset.map(process_path, num_parallel_calls=AUTOTUNE)

    return train_dataset, test_dataset, val_dataset


def prepare_for_training(ds, cache, repeat=1):
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()

    ds = ds.repeat(repeat)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds


def get_ds():
    train_ds, test_ds, val_ds = get_datasets_unprepared()
    train_ds = prepare_for_training(train_ds, cache=TRAIN_CACHE_FILE)
    test_ds = prepare_for_training(test_ds, cache=TEST_CACHE_FILE)
    val_ds = prepare_for_training(val_ds, cache=VAL_CACHE_FILE)
    return train_ds, test_ds, val_ds

