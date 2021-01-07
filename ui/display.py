import pygame
import os
import numpy as np
from scipy.spatial.distance import pdist, squareform

from game.enums import Terrain, Resource, PlayerId, BuildingType
from game.enums import TILE_NEIGHBOURS, HARBOUR_CORNER_AND_EDGES

class Display(object):
    def __init__(self, game):
        self.game = game

        self.hexagon_side_len = 82.25 * 1.2
        self.hexagon_height = int(2 * self.hexagon_side_len)
        self.hexagon_width = int(np.sqrt(3) * self.hexagon_side_len)

        self.OUTER_BOARD_SCALE = 1.1
        self.outer_hexagon_side_len = self.hexagon_side_len * self.OUTER_BOARD_SCALE
        self.outer_hexagon_height = int(2 * self.outer_hexagon_side_len)
        self.outer_hexagon_width = int(np.sqrt(3) * self.outer_hexagon_side_len)

        self.token_dim = 55
        self.building_scale = 0.4
        self.building_height = int(151 * self.building_scale)
        self.building_width = int(129 * self.building_scale)

        screen_width, screen_height = 1500, 1200

        self.first_tile_pos = (250, 200)

        self.scaled_tile_pos = {}
        self.tile_pos = {}
        self.tile_pos[0] = self.first_tile_pos
        self.scaled_tile_pos[0] = self.first_tile_pos
        for i in range(1, 19):
            for j in range(i):
                TL = TILE_NEIGHBOURS[i].get("TL", None)
                TR = TILE_NEIGHBOURS[i].get("TR", None)
                L = TILE_NEIGHBOURS[i].get("L", None)
                if TL is not None and TL == j:
                    self.tile_pos[i] = (self.tile_pos[j][0] + self.hexagon_side_len *(np.sqrt(3)/2.0),
                                        self.tile_pos[j][1] + (3.0/2.0)*self.hexagon_side_len)
                    self.scaled_tile_pos[i] = (self.scaled_tile_pos[j][0] + self.outer_hexagon_side_len * (np.sqrt(3) / 2.0),
                                        self.scaled_tile_pos[j][1] + (3.0 / 2.0) * self.outer_hexagon_side_len)
                    break
                elif TR is not None and TR == j:
                    self.tile_pos[i] = (self.tile_pos[j][0] - self.hexagon_side_len *(np.sqrt(3)/2.0),
                                        self.tile_pos[j][1] + (3.0/2.0)*self.hexagon_side_len)
                    self.scaled_tile_pos[i] = (self.scaled_tile_pos[j][0] - self.outer_hexagon_side_len * (np.sqrt(3) / 2.0),
                                        self.scaled_tile_pos[j][1] + (3.0 / 2.0) * self.outer_hexagon_side_len)
                    break
                elif L is not None and L == j:
                    self.tile_pos[i] = (self.tile_pos[j][0] + self.hexagon_width,
                                        self.tile_pos[j][1])
                    self.scaled_tile_pos[i] = (self.scaled_tile_pos[j][0] + self.outer_hexagon_width,
                                        self.scaled_tile_pos[j][1])

        self.corner_pos = {}
        self.scaled_corner_pos = {}
        for corner in game.board.corners:
            tile = corner.adjacent_tiles[0]
            start_pos = [self.tile_pos[tile.id][0] + self.hexagon_width / 2.0,
                         self.tile_pos[tile.id][1] + self.hexagon_height / 2.0]
            scaled_start_pos = [self.scaled_tile_pos[tile.id][0] + self.outer_hexagon_width / 2.0,
                         self.scaled_tile_pos[tile.id][1] + self.outer_hexagon_height / 2.0]
            for key, t_corner in tile.corners.items():
                if corner == t_corner:
                    if key == "T":
                        start_pos[1] -= self.hexagon_height / 2.0
                        scaled_start_pos[1] -= self.outer_hexagon_height / 2.0
                    elif key == "B":
                        start_pos[1] += self.hexagon_height / 2.0
                        scaled_start_pos[1] += self.outer_hexagon_height / 2.0
                    elif key == "TR":
                        start_pos[0] += self.hexagon_width / 2.0
                        start_pos[1] -= self.hexagon_side_len / 2.0
                        scaled_start_pos[0] += self.outer_hexagon_width / 2.0
                        scaled_start_pos[1] -= self.outer_hexagon_side_len / 2.0
                    elif key == "TL":
                        start_pos[0] -= self.hexagon_width / 2.0
                        start_pos[1] -= self.hexagon_side_len / 2.0
                        scaled_start_pos[0] -= self.outer_hexagon_width / 2.0
                        scaled_start_pos[1] -= self.outer_hexagon_side_len / 2.0
                    elif key == "BR":
                        start_pos[0] += self.hexagon_width / 2.0
                        start_pos[1] += self.hexagon_side_len / 2.0
                        scaled_start_pos[0] += self.outer_hexagon_width / 2.0
                        scaled_start_pos[1] += self.outer_hexagon_side_len / 2.0
                    elif key == "BL":
                        start_pos[0] -= self.hexagon_width / 2.0
                        start_pos[1] += self.hexagon_side_len / 2.0
                        scaled_start_pos[0] -= self.outer_hexagon_width / 2.0
                        scaled_start_pos[1] += self.outer_hexagon_side_len / 2.0
                    self.corner_pos[corner.id] = (int(start_pos[0]), int(start_pos[1]))
                    self.scaled_corner_pos[corner.id] = (int(scaled_start_pos[0]), int(scaled_start_pos[1]))
                    break

        self.image_path = ["ui", "images"]
        self.terrain_image_paths = {
            Terrain.Desert: "tiles/desert.png",
            Terrain.Fields: "tiles/wheat.png",
            Terrain.Mountains: "tiles/ore.png",
            Terrain.Forest: "tiles/wood.png",
            Terrain.Pastures: "tiles/sheep.png",
            Terrain.Hills: "tiles/brick.png"
        }
        self.terrain_images = {key: pygame.transform.scale(pygame.image.load(os.path.join(*self.image_path, val)),
                            (self.hexagon_width, self.hexagon_height)) for key, val in self.terrain_image_paths.items()}

        self.token_image_paths = {
            i: "value_tokens/token_"+str(i)+".png" for i in [2,3,4,5,6,8,9,10,11,12]
        }
        self.token_images = {key: pygame.transform.scale(pygame.image.load(os.path.join(*self.image_path, val)),
                                                           (self.token_dim, self.token_dim)) for key, val in
                               self.token_image_paths.items()}
        self.harbour_image_paths = {
            Resource.Brick: "harbours/harbour_brick.png",
            Resource.Wheat: "harbours/harbour_wheat.png",
            Resource.Ore: "harbours/harbour_ore.png",
            Resource.Sheep: "harbours/harbour_sheep.png",
            Resource.Wood: "harbours/harbour_wood.png",
            None: "harbours/harbour_none.png"
        }
        self.harbour_images = {key: pygame.transform.scale(pygame.image.load(os.path.join(*self.image_path, val)),
                                                         (self.token_dim, self.token_dim)) for key, val in
                             self.harbour_image_paths.items()}
        self.settlement_image_paths = {
            PlayerId.White: "buildings/settlement_white.png",
            PlayerId.Blue: "buildings/settlement_blue.png",
            PlayerId.Red: "buildings/settlement_red.png",
            PlayerId.Orange: "buildings/settlement_orange.png"
        }
        self.settlement_images = {key: pygame.transform.scale(pygame.image.load(os.path.join(*self.image_path, val)),
                                                         (self.building_width, self.building_height)) for key, val in
                             self.settlement_image_paths.items()}
        self.city_image_paths = {
            PlayerId.White: "buildings/city_white.png",
            PlayerId.Blue: "buildings/city_blue.png",
            PlayerId.Red: "buildings/city_red.png",
            PlayerId.Orange: "buildings/city_orange.png"
        }
        self.city_images = {key: pygame.transform.scale(pygame.image.load(os.path.join(*self.image_path, val)),
                                                              (self.building_width, self.building_height)) for key, val
                                  in self.city_image_paths.items()}

        pygame.init()
        self.construct_outer_board_polygon()
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        self.BACKGROUND_COLOUR = (25, 105, 158)
        self.screen.fill(self.BACKGROUND_COLOUR)

    def construct_outer_board_polygon(self):
        base_positions = np.array([self.scaled_corner_pos[corner.id] for corner in self.game.board.corners \
                              if corner.adjacent_tiles_placed < 3])
        dists = squareform(pdist(base_positions))
        positions = []
        positions_added = []
        curr_pos_ind = 0
        while len(positions) != len(base_positions):
            positions.append(base_positions[curr_pos_ind])
            positions_added.append(curr_pos_ind)
            min_dist = np.inf
            min_dist_ind = -1
            for i in range(len(base_positions)):
                if i != curr_pos_ind and i not in positions_added:
                    if dists[curr_pos_ind, i] < min_dist:
                        min_dist_ind = i
                        min_dist = dists[curr_pos_ind, i]
            if min_dist_ind != -1:
                curr_pos_ind = min_dist_ind
        for i in range(len(positions)):
            positions[i][0] = positions[i][0] - 1.5*(self.outer_hexagon_width - self.hexagon_width)
            positions[i][1] = positions[i][1] - 2*(self.outer_hexagon_height - self.hexagon_height)
        self.outer_board_polygon = positions

    def render(self):
        self.screen.fill(self.BACKGROUND_COLOUR)
        self.render_board()

        pygame.display.update()

    def render_board(self):
        pygame.draw.polygon(self.screen, pygame.Color(241, 233, 161),
                            self.outer_board_polygon)

        self.render_harbours()

        for i, tile in enumerate(self.game.board.tiles):
            self.render_tile(tile, self.tile_pos[i][0], self.tile_pos[i][1])
            if tile.value != 7:
                self.render_token(tile.value, self.tile_pos[i][0] + (self.hexagon_width / 2.0) - (self.token_dim / 2.0),
                                  self.tile_pos[i][1] + (self.hexagon_height / 2.0) - (self.token_dim / 2.0))
        for corner in self.game.board.corners:
            self.render_corner(corner)

    def render_tile(self, tile, x, y):
        self.screen.blit(self.terrain_images[tile.terrain], (x, y))

    def render_token(self, value, x, y):
        self.screen.blit(self.token_images[value], (x, y))

    def render_corner(self, corner):
        pygame.draw.circle(self.screen, pygame.Color("blue"), self.corner_pos[corner.id], 5)
        if corner.building is not None:
            if corner.building.type == BuildingType.Settlement:
                self.screen.blit(self.settlement_images[corner.building.owner],
                                 (self.corner_pos[corner.id][0] - (self.building_width/2.0),
                                 self.corner_pos[corner.id][1] - (self.building_height/2.0)))
            elif corner.building.type == BuildingType.City:
                self.screen.blit(self.city_images[corner.building.owner],
                                 (self.corner_pos[corner.id][0] - (self.building_width / 2.0),
                                  self.corner_pos[corner.id][1] - (self.building_height / 2.0)))

    def render_harbours(self):
        for i, harbour in enumerate(self.game.board.harbours):
            h_info = HARBOUR_CORNER_AND_EDGES[i]
            tile = self.game.board.tiles[h_info[0]]
            if h_info[3] == "TL":
                c1 = tile.corners["TL"].id
                c1_back = tile.corners["BL"].id
                c2 = tile.corners["T"].id
            elif h_info[3] == "TR":
                c1 = tile.corners["T"].id
                c1_back = tile.corners["TL"].id
                c2 = tile.corners["TR"].id
            elif h_info[3] == "R":
                c1 = tile.corners["TR"].id
                c1_back = tile.corners["T"].id
                c2 = tile.corners["BR"].id
            elif h_info[3] == "BR":
                c1 = tile.corners["BR"].id
                c1_back = tile.corners["TR"].id
                c2 = tile.corners["B"].id
            elif h_info[3] == "BL":
                c1 = tile.corners["B"].id
                c1_back = tile.corners["BR"].id
                c2 = tile.corners["BL"].id
            elif h_info[3] == "L":
                c1 = tile.corners["BL"].id
                c1_back = tile.corners["B"].id
                c2 = tile.corners["TL"].id
            corner_1_pos = np.array(self.corner_pos[c1])
            corner_1_back_pos = np.array(self.corner_pos[c1_back])
            corner_2_pos = np.array(self.corner_pos[c2])
            harbour_pos = corner_1_pos + (corner_1_pos - corner_1_back_pos)
            pygame.draw.line(self.screen, pygame.Color("black"), corner_1_pos,
                             harbour_pos, 3)
            pygame.draw.line(self.screen, pygame.Color("black"), corner_2_pos,
                             harbour_pos, 3)
            self.screen.blit(self.harbour_images[harbour.resource], (harbour_pos[0] - self.token_dim/2.0,
                                                                     harbour_pos[1] - self.token_dim/2.0))