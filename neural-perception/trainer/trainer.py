from tqdm import tqdm
import numpy as np


class Trainer:
    def __init__(self, model, data_loader, config, logger):
        self.model = model
        self.logger = logger
        self.config = config
        self.data_loader = data_loader

    def train(self):
        for epoch in range(self.config.num_epochs):
            self.train_epoch()

    def train_epoch(self):
        loop = tqdm(range(self.config.num_episodes))
        losses = []
        accs = []

        for _ in loop:
            loss, acc = self.train_step()
            losses.append(loss)
            accs.append(acc)

        loss = np.mean(losses)
        acc = np.mean(accs)

        # TODO: configure logger,
        #       save summaries,
        #       save model

        # cur_it = self.model.global_step_tensor.eval(self.sess)
        # summaries_dict = {
        #     'loss': loss,
        #     'acc': acc,
        # }
        # self.logger.summarize(cur_it, summaries_dict=summaries_dict)
        # self.model.save(self.sess)

    def train_step(self):
        batch_imgs, batch_labels = self.data_loader.next_batch(self.config.batch_size)
        # self.data_loader.show_batch(batch_imgs, batch_labels)

        norm_batch_imgs = batch_imgs / 255.
        norm_batch_imgs = norm_batch_imgs[:1]

        d, a = self.model.call(norm_batch_imgs[:2])
        # print("d: %s, a: %s" % (str(d.numpy()[0][0]), str(a.numpy()[0][0])))

        # TODO: calculate loss and acc
        loss = None
        acc = None

        return loss, acc
