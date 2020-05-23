import tensorflow as tf

from data.data_loader import DataLoader
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
    main()
