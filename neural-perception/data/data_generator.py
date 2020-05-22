from gym_duckietown.simulator import *
from gym_duckietown.graphics import *
from numpy import save, load
from matplotlib import pyplot as plt
from scipy.stats import truncnorm


class DuckietownImager(Simulator):

    def __init__(self, num_imgs, **kwargs):
        Simulator.__init__(self, **kwargs)
        self.map_name = 'udem1'
        self.domain_rand = True
        self.draw_bbox = False
        self.full_transparency = True
        self.accept_start_angle_deg = 90
        self.pos_bounds_fac = 0.4  # factor in road tile size for bounds of distribution
        self.num_imgs = num_imgs
        self.images = np.zeros(shape=(num_imgs, *self.observation_space.shape), dtype=self.observation_space.dtype)
        self.labels = np.zeros(shape=(num_imgs, 2), dtype=np.float32)

    def produce_images(self):

        for i in range(self.num_imgs):
            obs = self.reset_own()
            dist = self.get_agent_info()['Simulator']['lane_position']['dist'] / self.road_tile_size
            dot_dir = self.get_agent_info()['Simulator']['lane_position']['dot_dir']
            self.images[i] = obs
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

    def reset_own(self):
        """
                Same as Simulator.reset() but own pose distribution (normal instead of uniform)
                Reset the simulation at the start of a new episode
                This also randomizes many environment parameters (domain randomization)
                """

        # Step count since episode start
        self.step_count = 0
        self.timestamp = 0.0

        # Robot's current speed
        self.speed = 0

        if self.randomize_maps_on_reset:
            map_name = np.random.choice(self.map_names)
            self._load_map(map_name)

        if self.domain_rand:
            self.randomization_settings = self.randomizer.randomize()

        # Horizon color
        # Note: we explicitly sample white and grey/black because
        # these colors are easily confused for road and lane markings
        if self.domain_rand:
            horz_mode = self.randomization_settings['horz_mode']
            if horz_mode == 0:
                self.horizon_color = self._perturb(BLUE_SKY_COLOR)
            elif horz_mode == 1:
                self.horizon_color = self._perturb(WALL_COLOR)
            elif horz_mode == 2:
                self.horizon_color = self._perturb([0.15, 0.15, 0.15], 0.4)
            elif horz_mode == 3:
                self.horizon_color = self._perturb([0.9, 0.9, 0.9], 0.4)
        else:
            self.horizon_color = BLUE_SKY_COLOR

        # Setup some basic lighting with a far away sun
        if self.domain_rand:
            light_pos = self.randomization_settings['light_pos']
        else:
            light_pos = [-40, 200, 100]

        ambient = self._perturb([0.50, 0.50, 0.50], 0.3)
        # XXX: diffuse is not used?
        diffuse = self._perturb([0.70, 0.70, 0.70], 0.3)
        from pyglet import gl
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_POSITION, (gl.GLfloat * 4)(*light_pos))
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_AMBIENT, (gl.GLfloat * 4)(*ambient))
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_DIFFUSE, (gl.GLfloat * 4)(0.5, 0.5, 0.5, 1.0))
        gl.glEnable(gl.GL_LIGHT0)
        gl.glEnable(gl.GL_LIGHTING)
        gl.glEnable(gl.GL_COLOR_MATERIAL)

        # Ground color
        self.ground_color = self._perturb(GROUND_COLOR, 0.3)

        # Distance between the robot's wheels
        self.wheel_dist = self._perturb(WHEEL_DIST)

        # Distance bewteen camera and ground
        self.cam_height = self._perturb(CAMERA_FLOOR_DIST, 0.08)

        # Angle at which the camera is rotated
        self.cam_angle = [self._perturb(CAMERA_ANGLE, 0.2), 0, 0]

        # Field of view angle of the camera
        self.cam_fov_y = self._perturb(CAMERA_FOV_Y, 0.2)

        # Camera offset for use in free camera mode
        self.cam_offset = np.array([0, 0, 0], dtype=float)

        # Create the vertex list for the ground/noise triangles
        # These are distractors, junk on the floor
        numTris = 12
        verts = []
        colors = []
        for _ in range(0, 3 * numTris):
            p = self.np_random.uniform(low=[-20, -0.6, -20], high=[20, -0.3, 20], size=(3,))
            c = self.np_random.uniform(low=0, high=0.9)
            c = self._perturb([c, c, c], 0.1)
            verts += [p[0], p[1], p[2]]
            colors += [c[0], c[1], c[2]]
        import pyglet
        self.tri_vlist = pyglet.graphics.vertex_list(3 * numTris, ('v3f', verts), ('c3f', colors))

        # Randomize tile parameters
        for tile in self.grid:
            rng = self.np_random if self.domain_rand else None
            # Randomize the tile texture
            tile['texture'] = Texture.get(tile['kind'], rng=rng)

            # Random tile color multiplier
            tile['color'] = self._perturb([1, 1, 1], 0.2)

        # Randomize object parameters
        for obj in self.objects:
            # Randomize the object color
            obj.color = self._perturb([1, 1, 1], 0.3)

            # Randomize whether the object is visible or not
            if obj.optional and self.domain_rand:
                obj.visible = self.np_random.randint(0, 2) == 0
            else:
                obj.visible = True

        # If the map specifies a starting tile
        if self.user_tile_start:
            # logger.info('using user tile start: %s' % self.user_tile_start)
            i, j = self.user_tile_start
            tile = self._get_tile(i, j)
            if tile is None:
                msg = 'The tile specified does not exist.'
                raise Exception(msg)
            # logger.debug('tile: %s' % tile)
        else:
            if self.start_tile is not None:
                tile = self.start_tile
            else:
                # Select a random drivable tile to start on
                tile_idx = self.np_random.randint(0, len(self.drivable_tiles))
                tile = self.drivable_tiles[tile_idx]

        # Keep trying to find a valid spawn position on this tile

        for _ in range(MAX_SPAWN_ATTEMPTS):
            i, j = tile['coords']

            # Choose a random position on this tile
            x = self.np_random.uniform(i, i + 1) * self.road_tile_size
            z = self.np_random.uniform(j, j + 1) * self.road_tile_size
            propose_pos = np.array([x, 0, z])

            # normal instead of uniform
            propose_angle = get_truncated_normal(mean=0, sd=360 / 32, low=-90, upp=90).rvs()
            propose_angle = np.deg2rad(propose_angle)
            if propose_angle < 0:
                propose_angle += 2 * np.pi

            p, t = self.closest_curve_point(propose_pos, propose_angle)

            # normal distribution instead of uniform
            propose_pos = self.new_position(p, t)

            # If this is too close to an object or not a valid pose, retry
            inconvenient = self._inconvenient_spawn(propose_pos)

            if inconvenient:
                # msg = 'The spawn was inconvenient.'
                # logger.warning(msg)
                continue

            invalid = not self._valid_pose(propose_pos, propose_angle, safety_factor=1.3)
            if invalid:
                # msg = 'The spawn was invalid.'
                # logger.warning(msg)
                continue

            # If the angle is too far away from the driving direction, retry
            try:
                lp = self.get_lane_pos2(propose_pos, propose_angle)
            except NotInLane:
                continue
            M = self.accept_start_angle_deg
            ok = -M < lp.angle_deg < +M
            if not ok:
                continue
            # Found a valid initial pose
            break
        else:
            msg = 'Could not find a valid starting pose after %s attempts' % MAX_SPAWN_ATTEMPTS
            raise Exception(msg)

        self.cur_pos = propose_pos
        self.cur_angle = propose_angle

        # logger.info('Starting at %s %s' % (self.cur_pos, self.cur_angle))

        # Generate the first camera image
        obs = self.render_obs()

        # Return first observation
        return obs

    def new_position(self, pos, tangent):
        tangent = tangent * self.road_tile_size * self.pos_bounds_fac
        t1, t2 = rot_y(90) @ tangent, rot_y(270) @ tangent
        p1, p2 = pos + t1, pos + t2
        u = get_truncated_normal().rvs()
        new_p = (1 - u) * p1 + u * p2
        if new_p[0] >= round(pos[0]) + 1:
            new_p[0] = round(pos[0]) + 1
        if new_p[2] >= round(pos[2]) + 1:
            new_p[2] = round(pos[0]) + 1
        return new_p


def load_data(set_no, path="./generated/"):
    return load(path + "data" + str(set_no) + ".npy"), load(path + "labels" + str(set_no) + ".npy")


def rot_y(deg):
    rad = np.deg2rad(deg)
    c = math.cos(rad)
    s = math.sin(rad)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])


def get_truncated_normal(mean=0.5, sd=1 / 4, low=0, upp=1):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


if __name__ == '__main__':
    # plt.hist(get_truncated_normal(mean=0, sd=360 / 32, low=-90, upp=90).rvs(10000))
    # plt.show()

    imgs = 30
    env = DuckietownImager(imgs)
    env.generate_and_save(imgs)
    # for j in range(imgs):
    #     plt.imshow(env.images[j])
    #     plt.show()
