import numpy as np

from ..simulator import Simulator


class DuckietownImager(Simulator):

    def __init__(self, **kwargs):
        self.map_name = 'udem1',
        self.domain_rand = True,
        self.draw_bbox = False,
        self.full_transparency = True,
        self.accept_start_angle_deg = 360,
        Simulator.__init__(self, **kwargs)

    def get_images(self, num_imgs):

        images = list()
        for _ in range(num_imgs):
            start = list(self.drivable_tiles[int(np.random.uniform(0, len(self.drivable_tiles)))]['coords'])
            self.user_tile_start = start
            self.reset()
            dist = self.get_agent_info()['Simulator']['lane_position']['dist']
            dot_dir = self.get_agent_info()['Simulator']['lane_position']['dot_dir']
            images.append((self.render(mode='rgb_array'), (dist, dot_dir)))

        return images



