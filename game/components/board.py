import numpy as np
import copy

from game.enums import TILE_NEIGHBOURS, PREV_CORNER_LOOKUP, PREV_EDGE_LOOKUP, CORNER_NEIGHBOURS_IN_TILE, \
    HARBOUR_CORNER_AND_EDGES
from game.enums import Terrain, Resource
from game.components.tile import Tile
from game.components.corner import Corner
from game.components.edge import Edge
from game.components.harbour import Harbour
from game.components.buildings import Building, BuildingType

"""
tile position -> inds
      0    1    2
   3    4    5    6
7    8    9    10    11
  12   13   14    15
     16   17   18
"""

class Board(object):
    def __init__(self, randomise_number_placement = True, fixed_terrain_placements = None,
                 fixed_number_order = None):
        self.DEFAULT_NUMBER_ORDER = [5, 2, 6, 3, 8, 10, 9, 12, 11, 4, 8, 10, 9, 4, 5, 6, 3, 11]
        self.NUMBER_PLACEMENT_INDS = [0, 3, 7, 12, 16, 17, 18, 15, 11, 6, 2, 1, 4, 8, 13, 14, 10, 5, 9]
        self.TERRAIN_TO_PLACE = [Terrain.Desert] + [Terrain.Hills] * 3 + [Terrain.Fields] * 4 + \
                                [Terrain.Forest] * 4 + [Terrain.Mountains] * 3 + [Terrain.Pastures] * 4
        self.HARBOURS_TO_PLACE = [Harbour(Resource.Ore, exchange_value=2, id=0), Harbour(Resource.Sheep, exchange_value=2, id=1),
                                  Harbour(Resource.Wheat, exchange_value=2, id=2), Harbour(Resource.Wood, exchange_value=2, id=3),
                                  Harbour(Resource.Brick, exchange_value=2, id=4), Harbour(None, exchange_value=3, id=5),
                                  Harbour(None, exchange_value=3, id=6), Harbour(None, exchange_value=3, id=7),
                                  Harbour(None, exchange_value=3, id=8)]

        self.randomise_number_placement = randomise_number_placement
        self.fixed_terrain_placements = fixed_terrain_placements
        self.fixed_number_order = fixed_number_order

        if fixed_terrain_placements is not None:
            assert np.array_equal([fixed_terrain_placements.count(terrain) == self.TERRAIN_TO_PLACE.count(terrain)
                for terrain in [Terrain.Hills, Terrain.Forest, Terrain.Fields, Terrain.Pastures, Terrain.Mountains]])

        if fixed_number_order is not None:
            assert np.array_equal([fixed_number_order.count(n) == self.DEFAULT_NUMBER_ORDER.count(n) for n in
                                   range(2, 13)])

        self.reset()
        self.build_adjacency_matrices()

    def validate_number_order(self, number_order, terrain_order):
        tile_vals = {}
        n_ind = 0
        for i in range(19):
            if terrain_order[self.NUMBER_PLACEMENT_INDS[i]] == Terrain.Desert:
                tile_vals[self.NUMBER_PLACEMENT_INDS[i]] = 7
            else:
                tile_vals[self.NUMBER_PLACEMENT_INDS[i]] = number_order[n_ind]
                n_ind += 1
        for i in range(19):
            if tile_vals[i] == 6 or tile_vals[i] == 8:
                for key in TILE_NEIGHBOURS[i]:
                    neighbour_ind = TILE_NEIGHBOURS[i][key]
                    if tile_vals[neighbour_ind] == 6 or tile_vals[neighbour_ind] == 8:
                        return False
        return True

    def reset(self):
        if self.fixed_terrain_placements is not None:
            terrain_order = copy.copy(self.fixed_terrain_placements)
        else:
            terrain_order = copy.copy(self.TERRAIN_TO_PLACE)
            np.random.shuffle(terrain_order)

        if self.fixed_number_order is not None:
            number_order = copy.copy(self.fixed_number_order)
        else:
            number_order = copy.copy(self.DEFAULT_NUMBER_ORDER)
            if self.randomise_number_placement:
                np.random.shuffle(number_order)
                while self.validate_number_order(number_order, terrain_order) == False:
                    np.random.shuffle(number_order)

        self.harbours = copy.copy(self.HARBOURS_TO_PLACE)
        np.random.shuffle(self.harbours)
        for harbour in self.harbours:
            harbour.corners = []

        self.tiles = tuple([Tile(terrain_order[i], -1, i) for i in range(19)])
        self.value_to_tiles = {}
        num_ind = 0
        for i in range(19):
            if self.tiles[self.NUMBER_PLACEMENT_INDS[i]].terrain == Terrain.Desert:
                self.tiles[self.NUMBER_PLACEMENT_INDS[i]].value = 7
                self.tiles[self.NUMBER_PLACEMENT_INDS[i]].contains_robber = True
                self.robber_tile = self.tiles[self.NUMBER_PLACEMENT_INDS[i]]
            else:
                self.tiles[self.NUMBER_PLACEMENT_INDS[i]].value = number_order[num_ind]
                self.value_to_tiles[number_order[num_ind]] = self.value_to_tiles.get(number_order[num_ind], []) + \
                    [self.tiles[self.NUMBER_PLACEMENT_INDS[i]]]
                num_ind += 1
        self.corners = tuple([Corner(id = i) for i in range(54)])
        self.edges = tuple([Edge(id = i) for i in range(72)])
        corner_ind = 0; edge_ind = 0

        for tile_ind in range(19):
            for corner_location in self.tiles[tile_ind].corners.keys():
                prev_info = PREV_CORNER_LOOKUP[corner_location]
                prev_tile_ind = None
                prev_corner_loc = None
                if len(prev_info) > 0:
                    for info in prev_info:
                        ind = TILE_NEIGHBOURS[tile_ind].get(info[0], None)
                        if ind is not None:
                            prev_tile_ind = ind
                            prev_corner_loc = info[1]
                            break
                if prev_tile_ind is None:
                    self.tiles[tile_ind].corners[corner_location] = self.corners[corner_ind]
                    corner_ind += 1
                else:
                    self.tiles[tile_ind].corners[corner_location] = self.tiles[prev_tile_ind].corners[prev_corner_loc]

            for edge_location in self.tiles[tile_ind].edges.keys():
                prev_info = PREV_EDGE_LOOKUP[edge_location]
                prev_tile_ind = None
                prev_edge_loc = None
                if len(prev_info) > 0:
                    ind = TILE_NEIGHBOURS[tile_ind].get(prev_info[0], None)
                    if ind is not None:
                        prev_tile_ind = ind
                        prev_edge_loc = prev_info[1]
                if prev_tile_ind is None:
                    self.tiles[tile_ind].edges[edge_location] = self.edges[edge_ind]
                    edge_ind += 1
                else:
                    self.tiles[tile_ind].edges[edge_location] = self.tiles[prev_tile_ind].edges[prev_edge_loc]

        for tile_ind in range(19):
            for corner_loc, corner in self.tiles[tile_ind].corners.items():
                for n_corner_loc in CORNER_NEIGHBOURS_IN_TILE[corner_loc].keys():
                    edge_loc = CORNER_NEIGHBOURS_IN_TILE[corner_loc][n_corner_loc]
                    edge = self.tiles[tile_ind].edges[edge_loc]
                    n_corner = self.tiles[tile_ind].corners[n_corner_loc]
                    corner_included = False
                    for z in range(corner.neighbours_placed):
                        if n_corner == corner.corner_neighbours[z][0]:
                            corner_included = True
                    if corner_included == False:
                        edge.corner_1 = corner
                        edge.corner_2 = n_corner
                        corner.insert_neighbour(n_corner, edge)
                corner.insert_adjacent_tile(self.tiles[tile_ind])

        for i, harbour in enumerate(self.harbours):
            h_info = HARBOUR_CORNER_AND_EDGES[i]
            tile = self.tiles[h_info[0]]
            corner_1 = tile.corners[h_info[1]]
            corner_2 = tile.corners[h_info[2]]
            edge = tile.edges[h_info[3]]

            corner_1.harbour = harbour
            corner_2.harbour = harbour
            edge.harbour = harbour

            harbour.corners.append(corner_1)
            harbour.corners.append(corner_2)
            harbour.edge = edge

    def build_adjacency_matrices(self):
        self.corner_adjacency_matrix = np.zeros((54, 54))
        self.corner_egde_identification_map = np.zeros((54, 54))
        for corner in self.corners:
            for n_corner in corner.corner_neighbours:
                if n_corner[0] is not None:
                    self.corner_adjacency_matrix[corner.id, n_corner[0].id] = 1.0
                    self.corner_egde_identification_map[corner.id, n_corner[0].id] = n_corner[1].id

    def insert_settlement(self, player, corner, initial_placement = False):
        if corner.can_place_settlement(player.id, initial_placement=initial_placement):
            building = Building(BuildingType.Settlement, player.id, corner)
            corner.insert_building(building)
            if corner.harbour is not None:
                player.harbours[corner.harbour.resource] = corner.harbour
            return building
        else:
            raise ValueError("Cannot place settlement here.")

    def insert_city(self, player, corner):
        if corner.building is not None and \
                (corner.building.type == BuildingType.Settlement and corner.building.owner == player):
            building = Building(BuildingType.City, player, corner)
            corner.insert_building(building)
            return building
        else:
            raise ValueError("Cannot place city here!")

    def insert_road(self, player, edge):
        if edge.can_place_road(player):
            edge.insert_road(player)
        else:
            raise ValueError("Cannot place road here!")

    def get_available_settlement_locations(self, player, initial_round=False):
        available_locations = np.zeros((len(self.corners),), dtype=np.int)
        if initial_round == False:
            if player.resources[Resource.Wood] > 0 and player.resources[Resource.Sheep] > 0 and \
                player.resources[Resource.Brick] > 0 and player.resources[Resource.Wheat] > 0:
                pass
            else:
                return available_locations
        for i, corner in self.corners:
            if corner.can_place_settlement(player.id, initial_placement=initial_round):
                available_locations[i] = 1
        return available_locations

    def get_available_city_locations(self, player):
        available_locations = np.zeros((len(self.corners),), dtype=np.int)
        if player.resources[Resource.Ore] >= 3 and player.resources[Resource.Wheat] >= 2:
            pass
        else:
            return available_locations
        for i, corner in self.corners:
            if corner.building is not None and (corner.building.type == BuildingType.Settlement \
                and corner.building.owner == player.id):
                available_locations[i] = 1
        return available_locations

    def get_available_road_locations(self, player, initial_round=False):
        available_locations = np.zeros((len(self.edges),), dtype=np.int)
        if initial_round == False:
            if player.resources[Resource.Wood] > 0 and player.resources[Resource.Brick] > 0:
                pass
            else:
                return available_locations
        for i, edge in self.edges:
            if edge.can_place_road(player.id):
                available_locations[i] = 1
        return available_locations

    def move_robber(self, tile):
        self.robber_tile.contains_robber = False
        tile.contains_robber = True
        self.robber_tile = tile