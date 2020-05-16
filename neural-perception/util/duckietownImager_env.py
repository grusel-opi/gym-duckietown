from gym_duckietown.simulator import *
from gym_duckietown.graphics import *
from numpy import save, load
from matplotlib import pyplot as plt


class DuckietownImager(Simulator):

    def __init__(self, num_imgs, **kwargs):
        Simulator.__init__(self, **kwargs)
        self.map_name = 'udem1'
        self.domain_rand = True
        self.draw_bbox = False
        self.full_transparency = True
        self.accept_start_angle_deg = 180
        self.num_imgs = num_imgs
        self.images = np.zeros(shape=(num_imgs, WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8)
        self.labels = np.zeros(shape=(num_imgs, 2), dtype=np.float32)

    def produce_images(self):

        for i in range(self.num_imgs):

            start = list(self.drivable_tiles[int(np.random.uniform(0, len(self.drivable_tiles)))]['coords'])
            self.user_tile_start = start
            self.reset()
            dist = self.get_agent_info()['Simulator']['lane_position']['dist'] / self.road_tile_size
            dot_dir = self.get_agent_info()['Simulator']['lane_position']['dot_dir']
            img = self.render(mode='rgb_array')
            self.images[i] = img
            self.labels[i] = np.array([dist, dot_dir])

    def generate_and_save(self, sets=30, path="./generated/"):

        for i in range(sets):

            self.produce_images()

            try:
                os.mkdir(path)
            except OSError:
                pass

            save(path + "data" + str(i) + ".npy", self.images)
            save(path + "labels" + str(i) + ".npy", self.labels)


def load_data(path="./generated/"):

    loaded_i = load(path + "data.npy")
    loaded_l = load(path + "labels.npy")

    for j in range(10):
        plt.figure(str(loaded_l[j]))
        plt.imshow(loaded_i[j])

    plt.show()
