import pygame
import os
import sys
import copy
import numpy as np
import torch
from scipy.spatial.distance import pdist, squareform
from tkinter import messagebox, Tk

from game.enums import Terrain, Resource, PlayerId, BuildingType, ActionTypes, DevelopmentCard
from game.enums import TILE_NEIGHBOURS, HARBOUR_CORNER_AND_EDGES

from ui.sftext.sftext import SFText

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'sftext/'))

def draw_polygon_alpha(surface, color, points):
    lx, ly = zip(*points)
    min_x, min_y, max_x, max_y = min(lx), min(ly), max(lx), max(ly)
    target_rect = pygame.Rect(min_x, min_y, max_x - min_x, max_y - min_y)
    shape_surf = pygame.Surface(target_rect.size, pygame.SRCALPHA)
    pygame.draw.polygon(shape_surf, color, [(x - min_x, y - min_y) for x, y in points])
    surface.blit(shape_surf, target_rect)

class Display(object):
    def __init__(self, game=None, env=None, interactive=False, debug_mode=False, policies=None, test=False):
        if game is None:
            if env is None:
                raise RuntimeError("Need to provide display with either game or env")
            self.env = env
        else:
            self.env = env
            self.game = game
        self.interactive = interactive
        self.debug_mode = debug_mode

        self.policies = policies

        self.hexagon_side_len = 82.25 * 1.0
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

        self.dice_scale = 0.5
        self.dice_height = int(111 * self.dice_scale)
        self.dice_width = int(109 * self.dice_scale)

        if self.debug_mode:
            screen_width, screen_height = 2200, 1100
        else:
            screen_width, screen_height = 1735, 1100

        self.first_tile_pos = (250, 300)

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
        self.robber_image = pygame.transform.scale(pygame.image.load(os.path.join(*self.image_path,
                                        "value_tokens/token_robber.png")), (self.token_dim, self.token_dim))
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
                            (self.building_width, self.building_height)) for key, val in self.city_image_paths.items()}
        self.dice_image_paths = {
            1: "dice/dice_1.png",
            2: "dice/dice_2.png",
            3: "dice/dice_3.png",
            4: "dice/dice_4.png",
            5: "dice/dice_5.png",
            6: "dice/dice_6.png"
        }
        self.dice_images = {key: pygame.transform.scale(pygame.image.load(os.path.join(*self.image_path, val)),
                            (self.dice_width, self.dice_height)) for key, val in self.dice_image_paths.items()}
        self.development_card_scale = 0.36
        self.development_card_width, self.development_card_height = int(368 * self.development_card_scale), \
                                                                    int(304 * self.development_card_scale)
        self.played_development_cards_properties = {
            "start": [1230, 85],
            "max_in_row": 4,
            "x_shift": 115,
            "y_shift": 40
        }

        self.play_development_cards_start_pos = {
            "start": [1000, 495],
            "x_shift": 135,
        }

        self.development_card_image_paths = {
            DevelopmentCard.Knight: "development_cards/development_knight.png",
            DevelopmentCard.Monopoly: "development_cards/development_monopoly.png",
            DevelopmentCard.RoadBuilding: "development_cards/development_roadbuilding.png",
            DevelopmentCard.YearOfPlenty: "development_cards/development_yearofplenty.png",
            DevelopmentCard.VictoryPoint: "development_cards/development_victorypoint.png"
        }
        self.development_card_images = {key: pygame.transform.scale(pygame.image.load(os.path.join(*self.image_path, val)),
                            (self.development_card_width, self.development_card_height)) for key, val in
                            self.development_card_image_paths.items()}

        self.top_menu = pygame.image.load(os.path.join(*self.image_path, "menu/top_header.png"))
        self.action_menu = pygame.image.load(os.path.join(*self.image_path, "menu/action_menu.png"))
        self.building_cost_menu = pygame.image.load(os.path.join(*self.image_path, "menu/building_cost_menu.png"))
        self.tick_image = pygame.image.load(os.path.join(*self.image_path, "menu/tick.png"))

        self.ai_play_image = pygame.transform.scale(
            pygame.image.load(os.path.join(*self.image_path, "menu/ai_go.png")), (150, 150))

        pygame.init()
        pygame.font.init()
        self.top_menu_font = pygame.font.SysFont('Arial', 45)
        self.count_font = pygame.font.SysFont('Arial', 18)
        self.thinking_font = pygame.font.SysFont('Arial', 36)
        self.construct_outer_board_polygon()
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("Settlers of Catan RL environment")
        self.BACKGROUND_COLOUR = (25, 105, 158)
        self.road_colours = {
            PlayerId.White: (255, 255, 255),
            PlayerId.Red: (255, 0, 0),
            PlayerId.Blue: (0, 0, 255),
            PlayerId.Orange: (255, 153, 51)
        }
        self.ROAD_WIDTH = 15
        self.CORNER_RADIUS = 5

        development_card_res_box_width = 45
        development_card_res_box_height = 38
        self.development_card_res_boxes = {
            Resource.Wood: [1440, 456, development_card_res_box_width, development_card_res_box_height],
            Resource.Brick: [1496, 456, development_card_res_box_width, development_card_res_box_height],
            Resource.Sheep: [1552, 456, development_card_res_box_width, development_card_res_box_height],
            Resource.Wheat: [1608, 456, development_card_res_box_width, development_card_res_box_height],
            Resource.Ore: [1664, 456, development_card_res_box_width, development_card_res_box_height]
        }

        self.harbour_trade_res_boxes = {
            Resource.Wood: [989, 767, development_card_res_box_width, development_card_res_box_height],
            Resource.Brick: [1050, 767, development_card_res_box_width, development_card_res_box_height],
            Resource.Sheep: [1111, 767, development_card_res_box_width, development_card_res_box_height],
            Resource.Wheat: [1172, 767, development_card_res_box_width, development_card_res_box_height],
            Resource.Ore: [1233, 767, development_card_res_box_width, development_card_res_box_height]
        }

        self.harbour_receive_res_boxes = {
            Resource.Wood: [989, 840, development_card_res_box_width, development_card_res_box_height],
            Resource.Brick: [1050, 840, development_card_res_box_width, development_card_res_box_height],
            Resource.Sheep: [1111, 840, development_card_res_box_width, development_card_res_box_height],
            Resource.Wheat: [1172, 840, development_card_res_box_width, development_card_res_box_height],
            Resource.Ore: [1233, 840, development_card_res_box_width, development_card_res_box_height]
        }

        harbour_circle_radius = 30
        self.harbour_select_circles = {
            Resource.Wood: [(1000, 696), harbour_circle_radius],
            Resource.Brick: [(1070, 696), harbour_circle_radius],
            Resource.Sheep: [(1140, 696), harbour_circle_radius],
            Resource.Wheat: [(1210, 696), harbour_circle_radius],
            Resource.Ore: [(1280, 696), harbour_circle_radius],
            None: [(1350, 696), harbour_circle_radius]
        }

        semi_circle_scale = 0.58
        semi_circle_img_width = int(235 * semi_circle_scale)
        semi_circle_img_height = int(400 * semi_circle_scale)
        self.trading_semi_circle_image_paths = {
            PlayerId.White: "menu/semicircle_white.png",
            PlayerId.Red: "menu/semicircle_red.png",
            PlayerId.Blue: "menu/semicircle_blue.png",
            PlayerId.Orange: "menu/semicircle_orange.png"
        }
        self.trading_semi_circle_images = {key: pygame.transform.scale(pygame.image.load(os.path.join(*self.image_path, val)),
                                          (semi_circle_img_width, semi_circle_img_height)) for key, val in
                                          self.trading_semi_circle_image_paths.items()}

        self.trade_player_resource_boxes = {
            Resource.Wood: [1390, 720, development_card_res_box_width, development_card_res_box_height],
            Resource.Brick: [1362, 767, development_card_res_box_width, development_card_res_box_height],
            Resource.Sheep: [1352, 818, development_card_res_box_width, development_card_res_box_height],
            Resource.Wheat: [1363, 865, development_card_res_box_width, development_card_res_box_height],
            Resource.Ore: [1387, 902, development_card_res_box_width, development_card_res_box_height]
        }

        self.trade_player_active_boxes = [
            [1450, 750, development_card_res_box_width, development_card_res_box_height],
            [1412, 810, development_card_res_box_width, development_card_res_box_height],
            [1465, 810, development_card_res_box_width, development_card_res_box_height],
            [1450, 867, development_card_res_box_width, development_card_res_box_height]
        ]

        self.receive_player_resource_boxes = {
            Resource.Wood: [1602, 723, development_card_res_box_width, development_card_res_box_height],
            Resource.Brick: [1630, 765, development_card_res_box_width, development_card_res_box_height],
            Resource.Sheep: [1627, 810, development_card_res_box_width, development_card_res_box_height],
            Resource.Wheat: [1615, 857, development_card_res_box_width, development_card_res_box_height],
            Resource.Ore: [1595, 900, development_card_res_box_width, development_card_res_box_height]
        }

        self.receive_player_active_boxes = [
            [1533, 750, development_card_res_box_width, development_card_res_box_height],
            [1520, 810, development_card_res_box_width, development_card_res_box_height],
            [1573, 810, development_card_res_box_width, development_card_res_box_height],
            [1533, 867, development_card_res_box_width, development_card_res_box_height]
        ]

        self.player_box_width = 37
        self.player_box_height = 37
        self.player_boxes = {
            PlayerId.White: [1390, 646, self.player_box_width, self.player_box_height],
            PlayerId.Orange: [1435, 646, self.player_box_width, self.player_box_height],
            PlayerId.Red: [1480, 646, self.player_box_width, self.player_box_height],
            PlayerId.Blue: [1525, 646, self.player_box_width, self.player_box_height]
        }


        self.resource_image_paths = {
            Resource.Wood: "resources/wood.png",
            Resource.Brick: "resources/brick.png",
            Resource.Sheep: "resources/sheep.png",
            Resource.Wheat: "resources/wheat.png",
            Resource.Ore: "resources/ore.png"
        }
        self.resource_images = {key: pygame.image.load(os.path.join(*self.image_path, val)) for key, val in
                                self.resource_image_paths.items()}

        self.game_log = ""
        self.game_log_target_rect = pygame.Rect(1140, 335, 560, 120)
        self.game_log_surface = pygame.Surface(self.game_log_target_rect.size)
        self.game_log_sftext = SFText(text=self.game_log, surface=self.game_log_surface,
                                      font_path=os.path.join(os.path.dirname(__file__), "sftext/example/resources"))


        self.screen.fill(self.BACKGROUND_COLOUR)

        self.reset()

        if self.interactive:
            self.run_event_loop(test=test)

    def reset(self):
        self.active_other_player = []
        self.active_receive_res = []
        self.active_trade_res = []
        self.active_harbour = []
        self.active_harbour_receive_res = []
        self.active_harbour_trade_res = []
        self.active_development_res_boxes = []
        self.game_log_sftext.text = ""
        self.game_log_sftext.parse_text()
        self.message_count = 0

    def update_game_log(self, message):
        self.message_count += 1
        color = self.road_colours[message["player_id"]]
        message_to_add = "{style}{color "+str(color)+"}"+str(self.message_count) + ". " + message["text"] + "\n"
        self.game_log_sftext.text = message_to_add + self.game_log_sftext.text
        self.game_log_sftext.parse_text()

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
        self.game_log_sftext.post_update()
        pygame.event.pump()

    def render_game_log(self):
        self.game_log_surface.fill((57, 98, 137))
        self.game_log_sftext.on_update()
        self.screen.blit(self.game_log_surface, self.game_log_target_rect)

    def render_board(self):
        pygame.draw.polygon(self.screen, pygame.Color(241, 233, 161),
                            self.outer_board_polygon)

        self.render_harbours()

        for i, tile in enumerate(self.game.board.tiles):
            self.render_tile(tile, self.tile_pos[i][0], self.tile_pos[i][1])
            if tile.contains_robber:
                self.render_robber(self.tile_pos[i][0] + (self.hexagon_width / 2.0) - (self.token_dim / 2.0),
                                   self.tile_pos[i][1] + (self.hexagon_height / 2.0) + (self.token_dim / 2.0))
            if tile.value != 7:
                self.render_token(tile.value, self.tile_pos[i][0] + (self.hexagon_width / 2.0) - (self.token_dim / 2.0),
                                  self.tile_pos[i][1] + (self.hexagon_height / 2.0) - (self.token_dim / 2.0))
        for edge in self.game.board.edges:
            self.render_edge(edge)
        for corner in self.game.board.corners:
            self.render_corner(corner)

        self.render_action_menu()
        self.render_top_menu()

        self.render_development_card_res_boxes()
        self.render_harbour_exchange_images()
        self.render_harbour_res_boxes()
        self.render_trading()
        self.render_longest_road_largest_army()
        self.render_game_log()

        if self.debug_mode:
            self.render_debug_screen()

    def render_debug_screen(self):
        if self.game.must_respond_to_trade:
            player = self.game.players[self.game.proposed_trade["target_player"]]
        else:
            player = self.game.players[self.game.players_go]
        self.screen.blit(self.resource_images[Resource.Wood], (1950, 30))
        self.screen.blit(self.resource_images[Resource.Brick], (2000, 30))
        self.screen.blit(self.resource_images[Resource.Sheep], (2050, 30))
        self.screen.blit(self.resource_images[Resource.Wheat], (2100, 30))
        self.screen.blit(self.resource_images[Resource.Ore], (2150, 30))
        next_p_text = self.count_font.render("Next", False, (0,0,0))
        self.screen.blit(next_p_text, (1735, 105))
        pygame.draw.rect(self.screen, self.road_colours[player.inverse_player_lookup["next"]], (1825, 100, 30, 30))
        next_actual_text = self.count_font.render("Actual", False, (0,0,0))
        self.screen.blit(next_actual_text, (1885, 75))
        next_min_text = self.count_font.render("Min", False, (0,0,0))
        self.screen.blit(next_min_text, (1885, 105))
        next_max_text = self.count_font.render("Max", False, (0,0,0))
        self.screen.blit(next_max_text, (1885, 135))
        start_x = 1975; x_diff = 50; y1 = 75; y2 = 105; y3 = 135
        next_player = self.game.players[player.inverse_player_lookup["next"]]
        for res in [Resource.Wood, Resource.Brick, Resource.Sheep, Resource.Wheat, Resource.Ore]:
            actual_text = str(next_player.resources[res])
            self.screen.blit(self.count_font.render(actual_text, False, (255,255,255)), (start_x, y1))
            min_text = str(player.opponent_min_res["next"][res])
            self.screen.blit(self.count_font.render(min_text, False, (255,255,255)), (start_x, y2))
            max_text = str(player.opponent_max_res["next"][res])
            self.screen.blit(self.count_font.render(max_text, False, (255,255,255)), (start_x, y3))
            start_x += x_diff

        self.screen.blit(self.count_font.render("Next Next", False, (0,0,0)), (1720, 255))
        pygame.draw.rect(self.screen, self.road_colours[player.inverse_player_lookup["next_next"]], (1825, 250, 30, 30))
        self.screen.blit(self.count_font.render("Actual", False, (0,0,0)), (1885, 225))
        self.screen.blit(self.count_font.render("Min", False, (0,0,0)), (1885, 255))
        self.screen.blit(self.count_font.render("Max", False, (0,0,0)), (1885, 285))
        start_x = 1975; x_diff = 50; y1 = 225; y2 = 255; y3 = 285
        next_player = self.game.players[player.inverse_player_lookup["next_next"]]
        for res in [Resource.Wood, Resource.Brick, Resource.Sheep, Resource.Wheat, Resource.Ore]:
            actual_text = str(next_player.resources[res])
            self.screen.blit(self.count_font.render(actual_text, False, (255, 255, 255)), (start_x, y1))
            min_text = str(player.opponent_min_res["next_next"][res])
            self.screen.blit(self.count_font.render(min_text, False, (255, 255, 255)), (start_x, y2))
            max_text = str(player.opponent_max_res["next_next"][res])
            self.screen.blit(self.count_font.render(max_text, False, (255, 255, 255)), (start_x, y3))
            start_x += x_diff

        self.screen.blit(self.count_font.render("N N N", False, (0, 0, 0)), (1720, 405))
        pygame.draw.rect(self.screen, self.road_colours[player.inverse_player_lookup["next_next_next"]], (1825, 400, 30, 30))
        self.screen.blit(self.count_font.render("Actual", False, (0, 0, 0)), (1885, 375))
        self.screen.blit(self.count_font.render("Min", False, (0, 0, 0)), (1885, 405))
        self.screen.blit(self.count_font.render("Max", False, (0, 0, 0)), (1885, 435))
        start_x = 1975; x_diff = 50; y1 = 375; y2 = 405; y3 = 435
        next_player = self.game.players[player.inverse_player_lookup["next_next_next"]]
        for res in [Resource.Wood, Resource.Brick, Resource.Sheep, Resource.Wheat, Resource.Ore]:
            actual_text = str(next_player.resources[res])
            self.screen.blit(self.count_font.render(actual_text, False, (255, 255, 255)), (start_x, y1))
            min_text = str(player.opponent_min_res["next_next_next"][res])
            self.screen.blit(self.count_font.render(min_text, False, (255, 255, 255)), (start_x, y2))
            max_text = str(player.opponent_max_res["next_next_next"][res])
            self.screen.blit(self.count_font.render(max_text, False, (255, 255, 255)), (start_x, y3))
            start_x += x_diff

    def render_action_menu(self):
        if self.game.must_respond_to_trade:
            player = self.game.players[self.game.proposed_trade["target_player"]]
        else:
            player = self.game.players[self.game.players_go]
        self.screen.blit(self.action_menu, (843, 145))
        if self.game.die_1 is not None:
            self.screen.blit(self.dice_images[self.game.die_1], (890, 248))
            self.screen.blit(self.dice_images[self.game.die_2], (960, 248))
            total = int(self.game.die_1 + self.game.die_2)
            if total == 6 or total == 8:
                colour = (255, 0, 0)
            else:
                colour = (0, 0, 0)
            total = "(" + str(total) + ")"
            dice_sum = self.top_menu_font.render(total, False, colour)
            self.screen.blit(dice_sum, (1035, 248))
        if self.game.dice_rolled_this_turn:
            self.screen.blit(self.tick_image, (858, 325))
        card_pos = copy.copy(self.play_development_cards_start_pos["start"])
        shift = self.play_development_cards_start_pos["x_shift"]
        for card in [DevelopmentCard.Knight, DevelopmentCard.VictoryPoint, DevelopmentCard.RoadBuilding,
                     DevelopmentCard.YearOfPlenty, DevelopmentCard.Monopoly]:
            self.screen.blit(self.development_card_images[card], (card_pos[0], card_pos[1]))
            text = "x " +str(player.hidden_cards.count(card))
            card_count = self.count_font.render(text, False, (255, 255, 255))
            self.screen.blit(card_count, (card_pos[0] + shift - 15, card_pos[1] + 105))
            card_pos[0] += shift

        self.screen.blit(self.ai_play_image, (30, self.screen.get_height()-180))

    def render_top_menu(self):
        if self.game.players_need_to_discard:
            player = self.game.players[self.game.players_to_discard[0]]
        elif self.game.must_respond_to_trade:
            player = self.game.players[self.game.proposed_trade["target_player"]]
        else:
            player = self.game.players[self.game.players_go]
        pygame.draw.rect(self.screen, self.road_colours[player.id], (221, 21, 170, 90))
        self.screen.blit(self.top_menu, (0,0))
        vps = player.victory_points
        vp_text = self.top_menu_font.render(str(int(vps)), False, (0,0,0))
        self.screen.blit(vp_text, (267, 109))

        wood_text = self.top_menu_font.render(str(int(player.resources[Resource.Wood])), False, (0,0,0))
        self.screen.blit(wood_text, (445,109))

        brick_text = self.top_menu_font.render(str(int(player.resources[Resource.Brick])), False, (0, 0, 0))
        self.screen.blit(brick_text, (625, 109))

        sheep_text = self.top_menu_font.render(str(int(player.resources[Resource.Sheep])), False, (0, 0, 0))
        self.screen.blit(sheep_text, (791, 109))

        wheat_text = self.top_menu_font.render(str(int(player.resources[Resource.Wheat])), False, (0, 0, 0))
        self.screen.blit(wheat_text, (965, 109))

        ore_text = self.top_menu_font.render(str(int(player.resources[Resource.Ore])), False, (0, 0, 0))
        self.screen.blit(ore_text, (1139, 106))

        wood_count = self.count_font.render("x"+str(self.game.resource_bank[Resource.Wood]), False, (255, 255, 255))
        self.screen.blit(wood_count, (428, 23))

        brick_count = self.count_font.render("x" + str(self.game.resource_bank[Resource.Brick]), False, (255, 255, 255))
        self.screen.blit(brick_count, (599, 23))

        sheep_count = self.count_font.render("x" + str(self.game.resource_bank[Resource.Sheep]), False, (255, 255, 255))
        self.screen.blit(sheep_count, (770, 23))

        wheat_count = self.count_font.render("x" + str(self.game.resource_bank[Resource.Wheat]), False, (255, 255, 255))
        self.screen.blit(wheat_count, (940, 23))

        ore_count = self.count_font.render("x" + str(self.game.resource_bank[Resource.Ore]), False, (255, 255, 255))
        self.screen.blit(ore_count, (1110, 23))

        self.screen.blit(self.building_cost_menu, (830, 965))

        x_pos = self.played_development_cards_properties["start"][0]
        y_pos = self.played_development_cards_properties["start"][1]
        row_count = 0
        for card in player.visible_cards:
            self.screen.blit(self.development_card_images[int(card)], (x_pos, y_pos))
            row_count += 1
            if row_count == self.played_development_cards_properties["max_in_row"]:
                x_pos = self.played_development_cards_properties["start"][0]
                y_pos += self.played_development_cards_properties["y_shift"]
                row_count = 0
            else:
                x_pos += self.played_development_cards_properties["x_shift"]

    def render_longest_road_largest_army(self):
        largest_army_text = self.count_font.render("Largest Army: ", False, (0, 0, 0))
        self.screen.blit(largest_army_text, (10, 185))
        longest_road_text = self.count_font.render("Longest Road: ", False, (0, 0, 0))
        self.screen.blit(longest_road_text, (10, 218))

        if self.game.largest_army is not None:
            pygame.draw.rect(self.screen, self.road_colours[self.game.largest_army["player"]],
                             (130, 187, 20, 20))
            army_count = self.game.largest_army["count"]
            count_text = self.count_font.render("("+str(army_count)+")", False, (0, 0, 0))
            self.screen.blit(count_text, (160, 185))

        if self.game.longest_road is not None:
            pygame.draw.rect(self.screen, self.road_colours[self.game.longest_road["player"]],
                             (130, 220, 20, 20))
            road_count = self.game.longest_road["count"]
            count_text_2 = self.count_font.render("("+str(road_count)+")", False, (0, 0, 0))
            self.screen.blit(count_text_2, (160, 218))


    def render_trading(self):
        self.screen.blit(self.trading_semi_circle_images[self.game.players_go], (1401, 708))
        if len(self.active_other_player) > 0:
            self.screen.blit(pygame.transform.flip(
                self.trading_semi_circle_images[self.active_other_player[0]], True, False), (1493, 708))
        elif self.game.must_respond_to_trade:
            self.screen.blit(pygame.transform.flip(
                self.trading_semi_circle_images[self.game.proposed_trade["target_player"]], True, False), (1493, 708))

        for player_id in [PlayerId.White, PlayerId.Red, PlayerId.Blue, PlayerId.Orange]:
            pygame.draw.rect(self.screen, self.road_colours[player_id], self.player_boxes[player_id])
            if self.game.must_respond_to_trade == False and len(self.active_other_player) > 0:
                if self.active_other_player[0] == player_id:
                    pygame.draw.rect(self.screen, (0, 0, 0), self.player_boxes[player_id], width=4)

        for i in range(len(self.active_trade_res)):
            res = self.active_trade_res[i]
            rect = self.trade_player_active_boxes[i]
            self.screen.blit(self.resource_images[res], (rect[0], rect[1]))
        for i in range(len(self.active_receive_res)):
            res = self.active_receive_res[i]
            rect = self.receive_player_active_boxes[i]
            self.screen.blit(self.resource_images[res], (rect[0], rect[1]))

    def render_harbour_exchange_images(self):
        self.screen.blit(self.harbour_images[Resource.Wood], (973, 670))
        self.screen.blit(self.harbour_images[Resource.Brick], (1043, 670))
        self.screen.blit(self.harbour_images[Resource.Sheep], (1113, 670))
        self.screen.blit(self.harbour_images[Resource.Wheat], (1183, 670))
        self.screen.blit(self.harbour_images[Resource.Ore], (1253, 670))
        self.screen.blit(self.harbour_images[None], (1323, 670))

    def render_tile(self, tile, x, y):
        self.screen.blit(self.terrain_images[tile.terrain], (x, y))

    def render_token(self, value, x, y):
        self.screen.blit(self.token_images[value], (x, y))

    def render_robber(self, x, y):
        self.screen.blit(self.robber_image, (x, y))

    def render_corner(self, corner):
        pygame.draw.circle(self.screen, pygame.Color("blue"), self.corner_pos[corner.id], self.CORNER_RADIUS)
        if corner.building is not None:
            if corner.building.type == BuildingType.Settlement:
                self.screen.blit(self.settlement_images[corner.building.owner],
                                 (self.corner_pos[corner.id][0] - (self.building_width/2.0),
                                 self.corner_pos[corner.id][1] - (self.building_height/2.0)))
            elif corner.building.type == BuildingType.City:
                self.screen.blit(self.city_images[corner.building.owner],
                                 (self.corner_pos[corner.id][0] - (self.building_width / 2.0),
                                  self.corner_pos[corner.id][1] - (self.building_height / 2.0)))

    def render_edge(self, edge):
        if edge.road is not None:
            colour = self.road_colours[edge.road]
            pygame.draw.line(self.screen, pygame.Color(colour), self.corner_pos[edge.corner_1.id],
                             self.corner_pos[edge.corner_2.id], self.ROAD_WIDTH)

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

    def render_development_card_res_boxes(self):
        for res in self.active_development_res_boxes:
            pygame.draw.rect(self.screen, (0, 0, 0), self.development_card_res_boxes[res], width=4)

    def render_harbour_res_boxes(self):
        for res in self.active_harbour_trade_res:
            pygame.draw.rect(self.screen, (0, 0, 0), self.harbour_trade_res_boxes[res], width=4)
        for res in self.active_harbour_receive_res:
            pygame.draw.rect(self.screen, (0, 0, 0), self.harbour_receive_res_boxes[res], width=4)
        if len(self.active_harbour) > 0:
            pygame.draw.circle(self.screen, (0, 0, 0),
                               self.harbour_select_circles[self.active_harbour[0]][0],
                               self.harbour_select_circles[self.active_harbour[0]][1], width=4)

    def draw_invisible_edges(self):
        self.invisible_edges = []
        for edge in self.game.board.edges:
            line = pygame.draw.line(self.screen, pygame.Color((0,0,0,255)), self.corner_pos[edge.corner_1.id],
                                    self.corner_pos[edge.corner_2.id], self.ROAD_WIDTH)
            self.invisible_edges.append(line)

    def draw_invisible_hexagons(self):
        self.invisible_hexagons = []
        self.invisible_hexagon_points = []
        for tile in self.game.board.tiles:
            points = []
            for c_id in ["T", "TL", "BL", "B", "BR", "TR"]:
                position = self.corner_pos[tile.corners[c_id].id]
                points.append((position[0], position[1]))
            hexagon = pygame.draw.polygon(self.screen, pygame.Color((0,0,0)), points)
            self.invisible_hexagons.append(hexagon)
            self.invisible_hexagon_points.append(points.copy())

    def initialise_AI(self):
        self.curr_hidden_states = {}
        for player_id in [PlayerId.White, PlayerId.Blue, PlayerId.Red, PlayerId.Orange]:
            if isinstance(self.policies[player_id], str):
                pass
            else:
                self.dummy_policy = self.policies[player_id]
                self.curr_hidden_states[player_id] =  (torch.zeros(1, self.dummy_policy.lstm_size, device=self.dummy_policy.dummy_param.device),
                                                       torch.zeros(1, self.dummy_policy.lstm_size, device=self.dummy_policy.dummy_param.device))

    def step_AI(self, deterministic=True):
        players_go = self.get_players_turn()
        if isinstance(self.policies[players_go], str):
            messagebox.showinfo('Error', "It is currently a human player's turn.")
            return False
        else:
            curr_obs = self.dummy_policy.obs_to_torch(self.env._get_obs())
            curr_hidden_state = self.curr_hidden_states[players_go]
            action_masks = self.dummy_policy.act_masks_to_torch(self.env.get_action_masks())

            if self.policies[players_go].policy_type == "neural_network":
                _, actions, _, hidden_states = self.policies[players_go].act(
                    curr_obs, curr_hidden_state, torch.ones(1, 1, device=self.dummy_policy.dummy_param.device),
                    action_masks, deterministic=deterministic
                )
                actions = self.dummy_policy.torch_act_to_np(actions)
            elif self.policies[players_go].policy_type == "forward_search":
                self.render_thinking_text()
                pygame.display.update()
                pygame.event.pump()
                curr_state = self.env.save_state()
                placing_initial_settlement = False
                if self.env.game.initial_settlements_placed[players_go] == 0:
                    placing_initial_settlement = True
                elif self.env.game.initial_settlements_placed[players_go] == 1 and self.env.game.initial_roads_placed[
                    players_go] == 1:
                    placing_initial_settlement = True
                actions, hidden_states = self.policies[players_go].act(
                    curr_obs, self.curr_hidden_states, curr_state, action_masks, initial_settlement=placing_initial_settlement
                )
                self.render_board()
            else:
                raise NotImplementedError
            self.curr_hidden_states[players_go] = hidden_states

            _, _, done, info = self.env.step(
                actions
            )
            self.update_game_log(info["log"])
            if done:
                if players_go == PlayerId.Orange:
                    winner = "Orange"
                elif players_go == PlayerId.Red:
                    winner = "Red"
                elif players_go == PlayerId.Blue:
                    winner = "Blue"
                elif players_go == PlayerId.White:
                    winner = "White"
                final_message = "Game over. {} player wins!".format(winner)
                messagebox.showinfo('Game over', final_message)
            return done

    def get_players_turn(self):
        if self.game.players_need_to_discard:
            player_id = self.game.players_to_discard[0]
        elif self.game.must_respond_to_trade:
            player_id = self.game.proposed_trade["target_player"]
        else:
            player_id = self.game.players_go
        return player_id

    def render_thinking_text(self):
        thinking_text = self.thinking_font.render("THINKING...", False, (255, 255, 255))
        self.screen.blit(thinking_text, (30, self.screen.get_height() - 200))

    def run_event_loop(self, test=False):
        run = True

        if self.policies is not None:
            self.initialise_AI()

        while run:
            pygame.time.delay(150)
            Tk().wm_withdraw()
            self.screen.fill(self.BACKGROUND_COLOUR)
            self.draw_invisible_edges()
            if self.game.can_move_robber:
                self.draw_invisible_hexagons()
            else:
                self.invisible_hexagons = []
                self.invisible_hexagon_points = []
            self.render_board()

            mouse_click = False
            over_corner = False
            over_edge = False

            if test:
                players_go = self.get_players_turn()
                if isinstance(self.policies[players_go], str):
                    pass
                else:
                    done = self.step_AI()
                    if done:
                        break

                    pygame.display.update()
                    self.game_log_sftext.post_update()
                    pygame.event.pump()
                    continue

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                elif event.type == pygame.MOUSEBUTTONUP:
                    mouse_click = True
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button <= 3:
                        pass
                    else:
                        mouse_pos = pygame.mouse.get_pos()
                        if self.game_log_target_rect.collidepoint(*mouse_pos):
                            self.game_log_sftext.on_mouse_scroll(event)

            players_go = self.game.players[self.game.players_go]
            mouse_pos = pygame.mouse.get_pos()
            for corner in self.game.board.corners:
                corner_pos = self.corner_pos[corner.id]
                if corner.building is None:
                    if (corner_pos[0] - mouse_pos[0])**2 + (corner_pos[1] - mouse_pos[1])**2 <= (2*self.CORNER_RADIUS)**2:
                        pygame.draw.circle(self.screen, pygame.Color("blue"), self.corner_pos[corner.id],
                                           2*self.CORNER_RADIUS)
                        over_corner = True
                        if mouse_click:
                            action = {
                                "type": ActionTypes.PlaceSettlement,
                                "corner": corner.id
                            }
                            valid_action, error = self.game.validate_action(action, check_player=True)
                            if valid_action:
                                action_log = self.game.apply_action(action)
                                self.update_game_log(action_log)
                                self.render_corner(corner)
                            else:
                                messagebox.showinfo('Error', error)

                elif corner.building.type == BuildingType.Settlement:
                    x1 = corner_pos[0] - self.building_width // 2
                    x2 = x1 + self.building_width
                    y1 = corner_pos[1] - self.building_height // 2
                    y2 = y1 + self.building_height
                    if mouse_pos[0] > x1 and mouse_pos[0] < x2 and mouse_pos[1] > y1 and mouse_pos[1] < y2:
                        pygame.draw.rect(self.screen, (255, 255, 255), (x1, y1, (x2-x1), (y2-y1)), width=4)
                        over_corner = True
                        if mouse_click:
                            action = {
                                "type": ActionTypes.UpgradeToCity,
                                "corner": corner.id
                            }
                            valid_action, error = self.game.validate_action(action, check_player=True)
                            if valid_action:
                                action_log = self.game.apply_action(action)
                                self.update_game_log(action_log)
                                self.render_corner(corner)
                            else:
                                messagebox.showinfo('Error', error)

            for i, edge in enumerate(self.game.board.edges):
                if edge.road is None:
                    if self.invisible_edges[i].collidepoint(mouse_pos) and over_corner == False:
                        pygame.draw.line(self.screen, pygame.Color((0, 0, 0)),
                                                self.corner_pos[edge.corner_1.id],
                                                self.corner_pos[edge.corner_2.id], self.ROAD_WIDTH)
                        over_edge = True
                        if mouse_click:
                            action = {
                                "type": ActionTypes.PlaceRoad,
                                "edge": edge.id
                            }
                            valid_action, error = self.game.validate_action(action, check_player=True)
                            if valid_action:
                                action_log = self.game.apply_action(action)
                                self.update_game_log(action_log)
                                self.render_edge(edge)
                            else:
                                messagebox.showinfo('Error', error)

            if mouse_pos[0] > 914 and mouse_pos[0] < 1093 and mouse_pos[1] > 323 and mouse_pos[1] < 375:
                if mouse_click:
                    action = {
                        "type": ActionTypes.RollDice
                    }
                    valid_action, error = self.game.validate_action(action, check_player=True)
                    if valid_action:
                        action_log = self.game.apply_action(action)
                        self.update_game_log(action_log)
                    else:
                        messagebox.showinfo('Error', error)
            elif mouse_pos[0] > 914 and mouse_pos[0] < 1093 and mouse_pos[1] > 382 and mouse_pos[1] < 433:
                if mouse_click:
                    action = {
                        "type": ActionTypes.EndTurn
                    }
                    valid_action, error = self.game.validate_action(action, check_player=True)
                    if valid_action:
                        action_log = self.game.apply_action(action)
                        self.update_game_log(action_log)
                    else:
                        messagebox.showinfo('Error', error)
            elif mouse_pos[0] > 924 and mouse_pos[0] < 1127 and mouse_pos[1] > 445 and mouse_pos[1] < 485:
                if mouse_click:
                    action = {
                        "type": ActionTypes.BuyDevelopmentCard
                    }
                    valid_action, error = self.game.validate_action(action, check_player=True)
                    if valid_action:
                        action_log = self.game.apply_action(action)
                        self.update_game_log(action_log)
                    else:
                        messagebox.showinfo('Error', error)
            elif mouse_pos[0] > 1005 and mouse_pos[0] < 1128 and mouse_pos[1] > 523 and mouse_pos[1] < 600:
                points = [(1005, 523), (1128, 523), (1128, 600), (1005, 600)]
                draw_polygon_alpha(self.screen, (255, 255, 255, 125), points)
                if mouse_click:
                    action = {
                        "type": ActionTypes.PlayDevelopmentCard,
                        "card": DevelopmentCard.Knight
                    }
                    valid_action, error = self.game.validate_action(action, check_player=True)
                    if valid_action:
                        action_log = self.game.apply_action(action)
                        self.update_game_log(action_log)
                    else:
                        messagebox.showinfo('Error', error)
            elif mouse_pos[0] > 1141 and mouse_pos[0] < 1260 and mouse_pos[1] > 523 and mouse_pos[1] < 600:
                points = [(1141, 523), (1260, 523), (1260, 600), (1141, 600)]
                draw_polygon_alpha(self.screen, (255, 255, 255, 125), points)
                if mouse_click:
                    action = {
                        "type": ActionTypes.PlayDevelopmentCard,
                        "card": DevelopmentCard.VictoryPoint
                    }
                    valid_action, error = self.game.validate_action(action, check_player=True)
                    if valid_action:
                        action_log = self.game.apply_action(action)
                        self.update_game_log(action_log)
                    else:
                        messagebox.showinfo('Error', error)
            elif mouse_pos[0] > 1276 and mouse_pos[0] < 1395 and mouse_pos[1] > 523 and mouse_pos[1] < 600:
                points = [(1276, 523), (1395, 523), (1395, 600), (1276, 600)]
                draw_polygon_alpha(self.screen, (255, 255, 255, 125), points)
                if mouse_click:
                    action = {
                        "type": ActionTypes.PlayDevelopmentCard,
                        "card": DevelopmentCard.RoadBuilding
                    }
                    valid_action, error = self.game.validate_action(action, check_player=True)
                    if valid_action:
                        action_log = self.game.apply_action(action)
                        self.update_game_log(action_log)
                    else:
                        messagebox.showinfo('Error', error)
            elif mouse_pos[0] > 1412 and mouse_pos[0] < 1531 and mouse_pos[1] > 523 and mouse_pos[1] < 600:
                points = [(1412, 523), (1531, 523), (1531, 600), (1412, 600)]
                draw_polygon_alpha(self.screen, (255, 255, 255, 125), points)
                if mouse_click:
                    if len(self.active_development_res_boxes) == 1:
                        resource_1 = self.active_development_res_boxes[0]
                        resource_2 = self.active_development_res_boxes[0]
                    elif len(self.active_development_res_boxes) == 2:
                        resource_1 = self.active_development_res_boxes[0]
                        resource_2 = self.active_development_res_boxes[1]
                    else:
                        messagebox.showinfo('Error', "No resources selected for year of plenty card")
                        continue
                    action = {
                        "type": ActionTypes.PlayDevelopmentCard,
                        "card": DevelopmentCard.YearOfPlenty,
                        "resource_1": resource_1,
                        "resource_2": resource_2
                    }
                    valid_action, error = self.game.validate_action(action, check_player=True)
                    if valid_action:
                        action_log = self.game.apply_action(action)
                        self.update_game_log(action_log)
                    else:
                        messagebox.showinfo('Error', error)
            elif mouse_pos[0] > 1547 and mouse_pos[0] < 1666 and mouse_pos[1] > 523 and mouse_pos[1] < 600:
                points = [(1547, 523), (1666, 523), (1666, 600), (1547, 600)]
                draw_polygon_alpha(self.screen, (255, 255, 255, 125), points)
                if mouse_click:
                    if len(self.active_development_res_boxes) == 0:
                        messagebox.showinfo('Error', "Must select a resource.")
                        continue
                    elif len(self.active_development_res_boxes) > 1:
                        messagebox.showinfo('Error', "Can only choose one resource with monopoly.")
                        continue
                    else:
                        resource = self.active_development_res_boxes[0]
                    action = {
                        "type": ActionTypes.PlayDevelopmentCard,
                        "card": DevelopmentCard.Monopoly,
                        "resource": resource
                    }
                    valid_action, error = self.game.validate_action(action, check_player=True)
                    if valid_action:
                        action_log = self.game.apply_action(action)
                        self.update_game_log(action_log)
                    else:
                        messagebox.showinfo('Error', error)
            elif mouse_pos[0] > 1440 and mouse_pos[0] < (1664 + 45) and mouse_pos[1] > 456 and mouse_pos[1] < 494:
                for res in [Resource.Wood, Resource.Brick, Resource.Sheep, Resource.Wheat, Resource.Ore]:
                    rect = self.development_card_res_boxes[res]
                    if mouse_pos[0] > rect[0] and mouse_pos[0] < rect[0] + rect[2] and mouse_pos[1] > rect[1] and \
                        mouse_pos[1] < rect[1] + rect[3]:
                        pygame.draw.rect(self.screen, (255, 255, 255), rect, width=4)
                        if mouse_click:
                            if res in self.active_development_res_boxes:
                                self.active_development_res_boxes.remove(res)
                            else:
                                if len(self.active_development_res_boxes) < 2:
                                    self.active_development_res_boxes.append(res)
                                else:
                                    messagebox.showinfo('Error', "No development card involves more than 2 resources")
                        break
            elif mouse_pos[0] > 989 and mouse_pos[0] < (1233 + 45) and mouse_pos[1] > 767 and mouse_pos[1] < 805:
                for res in [Resource.Wood, Resource.Brick, Resource.Sheep, Resource.Wheat, Resource.Ore]:
                    rect = self.harbour_trade_res_boxes[res]
                    if mouse_pos[0] > rect[0] and mouse_pos[0] < rect[0] + rect[2] and mouse_pos[1] > rect[1] and \
                            mouse_pos[1] < rect[1] + rect[3]:
                        pygame.draw.rect(self.screen, (255, 255, 255), rect, width=4)
                        if mouse_click:
                            if res in self.active_harbour_trade_res:
                                self.active_harbour_trade_res.remove(res)
                            else:
                                if len(self.active_harbour_trade_res) == 0:
                                    self.active_harbour_trade_res.append(res)
                                else:
                                    messagebox.showinfo('Error', "Can only select one resource at a time.")
            elif mouse_pos[0] > 989 and mouse_pos[0] < (1233 + 45) and mouse_pos[1] > 840 and mouse_pos[1] < 878:
                for res in [Resource.Wood, Resource.Brick, Resource.Sheep, Resource.Wheat, Resource.Ore]:
                    rect = self.harbour_receive_res_boxes[res]
                    if mouse_pos[0] > rect[0] and mouse_pos[0] < rect[0] + rect[2] and mouse_pos[1] > rect[1] and \
                            mouse_pos[1] < rect[1] + rect[3]:
                        pygame.draw.rect(self.screen, (255, 255, 255), rect, width=4)
                        if mouse_click:
                            if res in self.active_harbour_receive_res:
                                self.active_harbour_receive_res.remove(res)
                            else:
                                if len(self.active_harbour_receive_res) == 0:
                                    self.active_harbour_receive_res.append(res)
                                else:
                                    messagebox.showinfo('Error', "Can only select one resource at a time.")
            elif mouse_pos[0] > 970 and mouse_pos[0] < 1380 and mouse_pos[1] > 666 and mouse_pos[1] < 726:
                for res in [Resource.Wood, Resource.Brick, Resource.Sheep, Resource.Wheat, Resource.Ore, None]:
                    centre = self.harbour_select_circles[res][0]; radius = self.harbour_select_circles[res][1]
                    if (mouse_pos[0] - centre[0])**2 + (mouse_pos[1] - centre[1])**2 <= radius**2:
                        pygame.draw.circle(self.screen, (255, 255, 255), centre, radius, width=4)
                        if mouse_click:
                            if res in self.active_harbour:
                                self.active_harbour.remove(res)
                            else:
                                if len(self.active_harbour) == 0:
                                    self.active_harbour.append(res)
                                else:
                                    messagebox.showinfo('Error', "Can only select one harbour at a time.")
            elif mouse_pos[0] > 1058 and mouse_pos[0] < 1203 and mouse_pos[1] > 899 and mouse_pos[1] < 934:
                if mouse_click:
                    if len(self.active_harbour_trade_res) == 0:
                        messagebox.showinfo('Error', "Need to select a resource to exchange.")
                    elif len(self.active_harbour_receive_res) == 0:
                        messagebox.showinfo('Error', "Need to select a resource to receive.")
                    else:
                        action = {
                            "type": ActionTypes.ExchangeResource
                        }
                        if len(self.active_harbour) > 0:
                            action["harbour"] = self.active_harbour[0]
                            if self.active_harbour[0] is None:
                                action["exchange_rate"] = 3
                            else:
                                action["exchange_rate"] = 2
                        else:
                            action["exchange_rate"] = 4
                        action["desired_resource"] = self.active_harbour_receive_res[0]
                        action["trading_resource"] = self.active_harbour_trade_res[0]
                        valid_action, error = self.game.validate_action(action, check_player=True)
                        if valid_action:
                            action_log = self.game.apply_action(action)
                            self.update_game_log(action_log)
                        else:
                            messagebox.showinfo('Error', error)
            elif mouse_pos[0] > 1390 and mouse_pos[0] < 1562 and mouse_pos[1] > 646 and mouse_pos[1] < 683:
                for player_id in [PlayerId.White, PlayerId.Red, PlayerId.Orange, PlayerId.Blue]:
                    box = self.player_boxes[player_id]
                    if mouse_pos[0] > box[0] and mouse_pos[0] < (box[0] + box[2]) and mouse_pos[1] > box[1] and \
                        mouse_pos[1] < (box[1] + box[3]):
                        pygame.draw.rect(self.screen, (255, 255, 255), box, width=4)
                        if mouse_click:
                            if self.game.must_respond_to_trade:
                                messagebox.showinfo('Error', "Cannot alter a proposed trade. Accept or reject.")
                            else:
                                self.active_other_player = [player_id]
            elif mouse_pos[0] > 1457 and mouse_pos[0] < 1575 and mouse_pos[1] > 932 and mouse_pos[1] < 963:
                if mouse_click:
                    if self.game.must_respond_to_trade:
                        messagebox.showinfo('Error', "Cannot modify a proposed trade. Accept or reject")
                    else:
                        self.active_receive_res = []
                        self.active_trade_res = []
            elif mouse_pos[0] > 1585 and mouse_pos[0] < 1700 and mouse_pos[1] > 932 and mouse_pos[1] < 963:
                if mouse_click:
                    if self.game.players_need_to_discard == False:
                        messagebox.showinfo('Error', "No one needs to discard resources at the moment")
                    else:
                        action = {
                            "type": ActionTypes.DiscardResource,
                            "resources": copy.copy(self.active_trade_res)
                        }
                        valid_action, error = self.game.validate_action(action, check_player=True)
                        if valid_action:
                            action_log = self.game.apply_action(action)
                            self.update_game_log(action_log)
                            self.active_trade_res = []
                            self.active_receive_res = []
                        else:
                            messagebox.showinfo('Error', error)
            elif mouse_pos[0] > 1437 and mouse_pos[0] < 1565 and mouse_pos[1] > 687 and mouse_pos[1] < 721:
                if mouse_click:
                    if len(self.active_other_player) == 0:
                        messagebox.showinfo('Error', "You need to select a player to steal from")
                    else:
                        action = {
                            "type": ActionTypes.StealResource,
                            "target": self.active_other_player[0]
                        }
                        valid_action, error = self.game.validate_action(action, check_player=True)
                        if valid_action:
                            action_log = self.game.apply_action(action)
                            self.update_game_log(action_log)
                        else:
                            messagebox.showinfo('Error', error)
            elif mouse_pos[0] > 1577 and mouse_pos[0] < 1705 and mouse_pos[1] > 687 and mouse_pos[1] < 721:
                if mouse_click:
                    if self.game.must_respond_to_trade:
                        action = {
                            "type": ActionTypes.RespondToOffer,
                            "response": "reject"
                        }
                        valid_action, error = self.game.validate_action(action, check_player=True)
                        if valid_action:
                            action_log = self.game.apply_action(action)
                            self.update_game_log(action_log)
                        else:
                            messagebox.showinfo('Error', error)
            elif mouse_pos[0] > 1577 and mouse_pos[0] < 1705 and mouse_pos[1] > 648 and mouse_pos[1] < 682:
                if mouse_click:
                    if self.game.must_respond_to_trade:
                        action = {
                            "type": ActionTypes.RespondToOffer,
                            "response": "accept"
                        }
                        valid_action, error = self.game.validate_action(action, check_player=True)
                        if valid_action:
                            action_log = self.game.apply_action(action)
                            self.update_game_log(action_log)
                        else:
                            messagebox.showinfo('Error', error)
                    else:
                        if len(self.active_trade_res) == 0:
                            messagebox.showinfo('Error', "Must choose resources to trade.")
                        elif len(self.active_receive_res) == 0:
                            messagebox.showinfo('Error', "Must propose resources to receive.")
                        elif len(self.active_other_player) == 0:
                            messagebox.showinfo('Error', "Must choose a player to propose the trade to.")
                        elif self.active_other_player[0] == players_go.id:
                            messagebox.showinfo('Error', "Cannot trade with yourself.")
                        else:
                            action = {
                                "type": ActionTypes.ProposeTrade,
                                "player_proposing": players_go.id,
                                "player_proposing_res": self.active_trade_res,
                                "target_player": self.active_other_player[0],
                                "target_player_res": self.active_receive_res
                            }
                            valid_action, error = self.game.validate_action(action, check_player=True)
                            if valid_action:
                                action_log = self.game.apply_action(action)
                                self.update_game_log(action_log)
                            else:
                                messagebox.showinfo('Error', error)
            elif mouse_pos[0] > 30 and mouse_pos[0] < 180 and mouse_pos[1] > self.screen.get_height() - 180 and \
                mouse_pos[1] < self.screen.get_height() - 30:
                if mouse_click:
                    self.step_AI()
            else:
                for res in [Resource.Wood, Resource.Brick, Resource.Sheep, Resource.Wheat, Resource.Ore]:
                    box = self.trade_player_resource_boxes[res]
                    box_2 = self.receive_player_resource_boxes[res]
                    if mouse_pos[0] > box[0] and mouse_pos[0] < (box[0] + box[2]) and mouse_pos[1] > box[1] and \
                            mouse_pos[1] < (box[1] + box[3]):
                        pygame.draw.rect(self.screen, (255, 255, 255), box, width=4)
                        if mouse_click:
                            if self.game.must_respond_to_trade:
                                messagebox.showinfo('Error', "Cannot modify proposed trade. Accept or reject.")
                            else:
                                if len(self.active_trade_res) < self.game.max_trade_resources:
                                    self.active_trade_res.append(res)
                                else:
                                    messagebox.showinfo('Error', "Can only trade up to 4 resources at a time.")
                    elif mouse_pos[0] > box_2[0] and mouse_pos[0] < (box_2[0] + box_2[2]) and mouse_pos[1] > box_2[1] and \
                            mouse_pos[1] < (box_2[1] + box_2[3]):
                        pygame.draw.rect(self.screen, (255, 255, 255), box_2, width=4)
                        if mouse_click:
                            if self.game.must_respond_to_trade:
                                messagebox.showinfo('Error', "Cannot modify proposed trade. Accept or reject.")
                            else:
                                if len(self.active_receive_res) < self.game.max_trade_resources:
                                    self.active_receive_res.append(res)
                                else:
                                    messagebox.showinfo('Error', "Can only trade up to 4 resources at a time.")

            if self.game.can_move_robber:
                if over_edge == False and over_corner == False:
                    for z, hexagon in enumerate(self.invisible_hexagons):
                        if hexagon.collidepoint(mouse_pos):
                            draw_polygon_alpha(self.screen, (255, 255, 255, 125), self.invisible_hexagon_points[z])
                            if mouse_click:
                                action = {
                                    "type": ActionTypes.MoveRobber,
                                    "tile": self.game.board.tiles[z].id
                                }
                                valid_action, error = self.game.validate_action(action, check_player=True)
                                if valid_action:
                                    action_log = self.game.apply_action(action)
                                    self.update_game_log(action_log)
                                else:
                                    messagebox.showinfo('Error', error)


            pygame.display.update()
            self.game_log_sftext.post_update()
            pygame.event.pump()