import tensorflow as tf

from data.data_loader import DataLoader
from data import data_generator
from model.model import PoseRegress
from trainer.trainer import Trainer
from util.config import process_config
from util.dirs import create_dirs
from util.logger import Logger
from util.args import get_args


def main():
    print("TensorFlow version: {}".format(tf.__version__))
    print("Eager execution: {}".format(tf.executing_eagerly()))

    try:
        args = get_args()
        config = process_config(args.config)
    except:
        print("missing or invalid arguments")
        exit(0)

    create_dirs([config.summary_dir, config.checkpoint_dir])
    data_loader = DataLoader(config)
    model = PoseRegress(config)

    # TODO: integrate the Logger
    # logger = Logger(config)

    # TODO: update the logger in the trainer
    trainer = Trainer(model, data_loader, config, logger=None)

    # TODO: Support the model to save and load
    # model.load(sess)

    trainer.train()


if __name__ == '__main__':

    observations = 16
    batches = 1

    model = PoseRegress(config=None)
    data, _ = data_generator.get_in_ram_sample(observations)
    data = data / 255.
    batch_size = int(len(data) / batches)

    results = []

    print("batch size: %d" % batch_size)

    for b in range(batches):
        print("batch no. %d" % b)
        batch = data[b*batch_size:b*(batch_size + 1)]
        results.append(model(batch))

    for r in results:
        print(r)
