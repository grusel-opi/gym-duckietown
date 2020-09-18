from gym_duckietown.simulator import *
from gym_duckietown.graphics import *
from gym_duckietown.envs import DuckietownEnv
from numpy import load
from matplotlib import pyplot as plt
from scipy.stats import truncnorm
from neural_perception.util import OwnLanePosition
import os
import sys


class ControlledDuckietownImager(DuckietownEnv):
    def __init__(self, **kwargs):
        DuckietownEnv.__init__(self, **kwargs)
        self.max_steps = math.inf
        self.map_name = 'udem1'
        self.domain_rand = True
        self.draw_bbox = False
        self.full_transparency = True
        self.set_size = 100
        self.path = "../../datasets/generated_controlled_dist_to_edge_better_in_cm/"
        self.k_p = 1
        self.k_d = 1
        self.action_speed = 0.2
        self.images = np.zeros(shape=(self.set_size, *self.observation_space.shape),
                               dtype=self.observation_space.dtype)
        self.labels = np.zeros(shape=(self.set_size, 2), dtype=np.float32)

    def get_lane_pos(self, pos, angle):
        point, tangent = self.closest_curve_point(pos, angle)
        if point is None:
            msg = 'Point not in lane: %s' % pos
            raise NotInLane(msg)

        assert point is not None

        dirVec = get_dir_vec(angle)
        dotDir = np.dot(dirVec, tangent)
        dotDir = max(-1, min(1, dotDir))

        posVec = pos - point
        upVec = np.array([0, 1, 0])
        rightVec = np.cross(tangent, upVec)
        signedDist = np.dot(posVec, rightVec)
        dist_to_road_edge = 0.25 * self.road_tile_size - signedDist
        angle_rad = math.acos(dotDir)

        if np.dot(dirVec, rightVec) < 0:
            angle_rad *= -1

        angle_deg = np.rad2deg(angle_rad)

        return OwnLanePosition(dist=signedDist,
                               dist_to_edge=dist_to_road_edge,
                               dot_dir=dotDir,
                               angle_deg=angle_deg,
                               angle_rad=angle_rad)

    # # @Override
    # def get_lane_pos2(self, pos, angle):
    #     point, tangent = self.closest_curve_point(pos, angle)
    #     if point is None:
    #         msg = 'Point not in lane: %s' % pos
    #         raise NotInLane(msg)
    #
    #     assert point is not None
    #
    #     track_width = 0.4
    #     a = track_width / 2
    #     rot = Rotation.from_rotvec(np.radians(-90) * np.array([0, 1, 0]))
    #     rot_tangent = rot.apply(tangent * a)
    #     new_point = point + rot_tangent
    #
    #     dir_vec = get_dir_vec(angle)
    #     dot_dir = np.dot(dir_vec, tangent)
    #     dot_dir = max(-1, min(1, dot_dir))
    #
    #     # Compute the signed distance to the curve
    #     # Right of the curve is negative, left is positive
    #     new_pos_vec = new_point - pos
    #     pos_vec = pos - point
    #     up_vec = np.array([0, 1, 0])
    #     right_vec = np.cross(tangent, up_vec)
    #     signed_dist_egde = np.dot(new_pos_vec, right_vec)
    #     signed_dist_center = np.dot(pos_vec, right_vec)
    #
    #     # Compute the signed angle between the direction and curve tangent
    #     # Right of the tangent is negative, left is positive
    #     angle_rad = math.acos(dot_dir)
    #
    #     if np.dot(dir_vec, right_vec) < 0:
    #         angle_rad *= -1
    #
    #     angle_deg = np.rad2deg(angle_rad)
    #
    #     return OwnLanePosition(dist=signed_dist_center, dist_to_edge=signed_dist_egde, dot_dir=dot_dir,
    #                            angle_deg=angle_deg,
    #                            angle_rad=angle_rad)

    def produce_images(self, n=10):
        obs = self.reset()
        for i in range(self.set_size):
            for _ in range(n):  # do n steps for every image
                try:
                    lp = self.get_lane_pos(self.cur_pos, self.cur_angle)
                except NotInLane:
                    self.reset()
                    continue
                distance_to_road_center = lp.dist
                angle_from_straight_in_rads = lp.angle_rad
                steering = self.k_p * distance_to_road_center + self.k_d * angle_from_straight_in_rads
                action = np.array([self.action_speed, steering])
                obs, reward, done, info = self.step(action)
                if done:
                    print("***DONE***")
                    self.reset()

            if lp.dist_to_edge < 0:
                continue
            else:
                self.images[i] = obs
                self.labels[i] = np.array([lp.dist_to_edge * 100, lp.angle_deg])

    def generate_and_save(self, num_sets):
        try:
            os.mkdir(self.path)
        except OSError:
            if os.path.isdir(self.path):
                pass
            else:
                return
        for _ in range(num_sets):
            self.produce_images()
            for i in range(self.set_size):
                plt.imsave(self.path + str(self.labels[i]) + '.png', self.images[i])


class RandomDuckietownImager(Simulator):

    def __init__(self, **kwargs):
        Simulator.__init__(self, **kwargs)
        self.map_name = 'udem1'
        self.domain_rand = True
        self.draw_bbox = False
        self.full_transparency = True
        self.accept_start_angle_deg = 90
        self.angle_variance = 20
        self.pos_bounds_fac = 0.4  # factor in road tile size for bounds of distribution
        self.set_size = 1000
        self.path = "../../generated_small/"
        self.images = np.zeros(shape=(self.set_size, *self.observation_space.shape), dtype=self.observation_space.dtype)
        self.labels = np.zeros(shape=(self.set_size, 2), dtype=np.float32)

    def produce_images(self):
        for i in range(self.set_size):
            obs = self.reset_own()
            dist = self.get_agent_info()['Simulator']['lane_position']['dist'] / self.road_tile_size
            dot_dir = self.get_agent_info()['Simulator']['lane_position']['dot_dir']
            self.images[i] = obs
            self.labels[i] = np.array([dist, dot_dir])

    def generate_and_save(self, sets=30):
        try:
            os.mkdir(self.path)
        except OSError:
            if os.path.isdir(self.path):
                pass
            else:
                return
        for _ in range(sets):
            self.produce_images()
            for i in range(self.set_size):
                plt.imsave(self.path + str(self.labels[i]) + '.png', self.images[i])

    def load_data(self, set_no):
        return load(self.path + "data" + str(set_no) + ".npy"), load(self.path + "labels" + str(set_no) + ".npy")

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
            propose_angle = get_truncated_normal(mean=0,
                                                 sd=self.angle_variance,
                                                 low=-1 * self.accept_start_angle_deg,
                                                 upp=self.accept_start_angle_deg).rvs()
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


def get_in_ram_sample(num):
    env = RandomDuckietownImager(set_size=num)
    env.produce_images()
    return env.images, env.labels


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
    imgs = 60_000
    env = ControlledDuckietownImager()
    setsize = env.set_size
    sets = imgs // setsize
    print("Generating ", sets, " sets")
    env.generate_and_save(sets)
    sys.exit()
