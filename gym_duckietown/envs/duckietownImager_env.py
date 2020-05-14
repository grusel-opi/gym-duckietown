from ..simulator import *
from ..graphics import *


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
        # self.own_curves = [self.get_own_curves(t) for t in self.grid]

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

    # def get_lane_pos3(self, pos, angle):
    #     """
    #     Get the position of the agent relative to the closest edge of road
    #     """
    #
    #     point, tangent = self.closest_edge_point(pos, angle)
    #
    #     if point is None:
    #         msg = 'Point not in lane: %s' % pos
    #         raise NotInLane(msg)
    #
    #     assert point is not None
    #
    #     # Compute the alignment of the agent direction with the curve tangent
    #     dirVec = get_dir_vec(angle)
    #     dotDir = np.dot(dirVec, tangent)
    #     dotDir = max(-1, min(1, dotDir))
    #
    #     # Compute the signed distance to the curve
    #     # Right of the curve is negative, left is positive
    #     posVec = pos - point
    #     upVec = np.array([0, 1, 0])
    #     rightVec = np.cross(tangent, upVec)
    #     signedDist = np.dot(posVec, rightVec)
    #
    #     return signedDist, dotDir
    #
    # def closest_edge_point(self, pos, angle=None):
    #     """
    #         Get the closest point to the edge of road to a given point
    #         Also returns the tangent at that point.
    #
    #         Returns None, None if not in a lane.
    #     """
    #
    #     i, j = self.get_grid_coords(pos)
    #     tile = self._get_tile(i, j)
    #
    #     if tile is None or not tile['drivable']:
    #         return None, None
    #
    #     curves = self._get_own_curve_for_tile(*tile['coords'])
    #
    #     # TODO: how to get closest cps
    #     cps = get_closest_own_bezier(curves)
    #
    #     t = bezier_closest(cps, pos)
    #     point = bezier_point(cps, t)
    #     tangent = bezier_tangent(cps, t)
    #
    #     return point, tangent
    #
    # def get_own_curves(self, tile):
    #     """
    #         Get our own Bezier curve control points for a given tile
    #     """
    #
    #     kind = tile['kind']
    #     angle = tile['angle']
    #     i, j = tile['coords']
    #
    #     # Each tile will have a unique set of control points,
    #     # Corresponding to each of its possible turns
    #     # TODO: make new coords
    #     if kind.startswith('straight'):
    #         pts = np.array([
    #             [
    #                 [-0.20, 0, -0.50],
    #                 [-0.20, 0, -0.25],
    #                 [-0.20, 0, 0.25],
    #                 [-0.20, 0, 0.50],
    #             ],
    #             [
    #                 [0.20, 0, 0.50],
    #                 [0.20, 0, 0.25],
    #                 [0.20, 0, -0.25],
    #                 [0.20, 0, -0.50],
    #             ]
    #         ]) * self.road_tile_size
    #
    #     elif kind == 'curve_left':
    #         pts = np.array([
    #             [
    #                 [-0.20, 0, -0.50],
    #                 [-0.20, 0, 0.00],
    #                 [0.00, 0, 0.20],
    #                 [0.50, 0, 0.20],
    #             ],
    #             [
    #                 [0.50, 0, -0.20],
    #                 [0.30, 0, -0.20],
    #                 [0.20, 0, -0.30],
    #                 [0.20, 0, -0.50],
    #             ]
    #         ]) * self.road_tile_size
    #
    #     elif kind == 'curve_right':
    #         pts = np.array([
    #             [
    #                 [-0.20, 0, -0.50],
    #                 [-0.20, 0, -0.20],
    #                 [-0.30, 0, -0.20],
    #                 [-0.50, 0, -0.20],
    #             ],
    #
    #             [
    #                 [-0.50, 0, 0.20],
    #                 [-0.30, 0, 0.20],
    #                 [0.30, 0, 0.00],
    #                 [0.20, 0, -0.50],
    #             ]
    #         ]) * self.road_tile_size
    #
    #     # Hardcoded all curves for 3way intersection
    #     elif kind.startswith('3way'):
    #         pts = np.array([
    #             [
    #                 [-0.20, 0, -0.50],
    #                 [-0.20, 0, -0.25],
    #                 [-0.20, 0, 0.25],
    #                 [-0.20, 0, 0.50],
    #             ],
    #             [
    #                 [-0.20, 0, -0.50],
    #                 [-0.20, 0, 0.00],
    #                 [0.00, 0, 0.20],
    #                 [0.50, 0, 0.20],
    #             ],
    #             [
    #                 [0.20, 0, 0.50],
    #                 [0.20, 0, 0.25],
    #                 [0.20, 0, -0.25],
    #                 [0.20, 0, -0.50],
    #             ],
    #             [
    #                 [0.50, 0, -0.20],
    #                 [0.30, 0, -0.20],
    #                 [0.20, 0, -0.20],
    #                 [0.20, 0, -0.50],
    #             ],
    #             [
    #                 [0.20, 0, 0.50],
    #                 [0.20, 0, 0.20],
    #                 [0.30, 0, 0.20],
    #                 [0.50, 0, 0.20],
    #             ],
    #             [
    #                 [0.50, 0, -0.20],
    #                 [0.30, 0, -0.20],
    #                 [-0.20, 0, 0.00],
    #                 [-0.20, 0, 0.50],
    #             ],
    #         ]) * self.road_tile_size
    #
    #     # Template for each side of 4way intersection
    #     elif kind.startswith('4way'):
    #         pts = np.array([
    #             [
    #                 [-0.20, 0, -0.50],
    #                 [-0.20, 0, 0.00],
    #                 [0.00, 0, 0.20],
    #                 [0.50, 0, 0.20],
    #             ],
    #             [
    #                 [-0.20, 0, -0.50],
    #                 [-0.20, 0, -0.25],
    #                 [-0.20, 0, 0.25],
    #                 [-0.20, 0, 0.50],
    #             ],
    #             [
    #                 [-0.20, 0, -0.50],
    #                 [-0.20, 0, -0.20],
    #                 [-0.30, 0, -0.20],
    #                 [-0.50, 0, -0.20],
    #             ]
    #         ]) * self.road_tile_size
    #     else:
    #         assert False, kind
    #
    #     # Rotate and align each curve with its place in global frame
    #     if kind.startswith('4way'):
    #         fourway_pts = []
    #         # Generate all four sides' curves,
    #         # with 3-points template above
    #         for rot in np.arange(0, 4):
    #             mat = gen_rot_matrix(np.array([0, 1, 0]), rot * math.pi / 2)
    #             pts_new = np.matmul(pts, mat)
    #             pts_new += np.array([(i + 0.5) * self.road_tile_size, 0, (j + 0.5) * self.road_tile_size])
    #             fourway_pts.append(pts_new)
    #
    #         fourway_pts = np.reshape(np.array(fourway_pts), (12, 4, 3))
    #         return fourway_pts
    #
    #     # Hardcoded each curve; just rotate and shift
    #     elif kind.startswith('3way'):
    #         threeway_pts = []
    #         mat = gen_rot_matrix(np.array([0, 1, 0]), angle * math.pi / 2)
    #         pts_new = np.matmul(pts, mat)
    #         pts_new += np.array([(i + 0.5) * self.road_tile_size, 0, (j + 0.5) * self.road_tile_size])
    #         threeway_pts.append(pts_new)
    #
    #         threeway_pts = np.array(threeway_pts)
    #         threeway_pts = np.reshape(threeway_pts, (6, 4, 3))
    #         return threeway_pts
    #
    #     else:
    #         mat = gen_rot_matrix(np.array([0, 1, 0]), angle * math.pi / 2)
    #         pts = np.matmul(pts, mat)
    #         pts += np.array([(i + 0.5) * self.road_tile_size, 0, (j + 0.5) * self.road_tile_size])
    #
    #     return pts
    #
    # def _get_own_curve_for_tile(self, i, j):
    #     i = int(i)
    #     j = int(j)
    #     if i < 0 or i >= self.grid_width:
    #         return None
    #     if j < 0 or j >= self.grid_height:
    #         return None
    #     return self.grid[j * self.grid_width + i]
    #
    # def get_closest_own_bezier(self, curves):
    #     pass
