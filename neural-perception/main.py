import tensorflow as tf
import sys

sys.path.append('.')

from .data.data_loader import DataLoader
from .model.model import PoseRegress
from .train.train import Train
from .util.config import process_config
from .util.dirs import create_dirs
from .util.logger import Logger
from .util.args import get_args


def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args()
        config = process_config(args.config)
    except:
        print("missing or invalid arguments")
        exit(0)

    # create the experiments dirs
    create_dirs([config.summary_dir, config.checkpoint_dir])
    # create tensorflow session
    sess = tf.Session()

    data_loader = DataLoader(config)
    print(data_loader.data)

    # create an instance of the model you want
    model = PoseRegress(config)
    # create tensorboard logger
    logger = Logger(sess, config)
    # create trainer and pass all the previous components to it
    trainer = Train(sess, model, data_loader, config, logger)
    # load model if exists
    model.load(sess)
    # here you train your model
    trainer.train()


if __name__ == '__main__':
    main()
