from game.enums import BuildingType

class Corner(object):
    def __init__(self, id):
        self.id = id

        self.neighbours_placed = 0
        self.adjacent_tiles_placed = 0
        self.corner_neighbours = [[None, None], [None, None], [None, None]]
        self.adjacent_tiles = [None, None, None]

        self.harbour = None
        self.building = None

    def __eq__(self, other):
        if self.id == other.id:
            return True
        else:
            return False

    def insert_building(self, building):
        self.building = building

    def can_place_settlement(self, player, initial_placement = False):
        player_roads = 0
        if self.building is not None:
            return False
        for corner in self.corner_neighbours:
            if corner[0] is not None:
                if corner[0].building is not None:
                    return False
                if corner[1].road is not None and corner[1].road == player:
                    player_roads += 1
        if initial_placement:
            return True
        if player_roads > 0:
            return True
        else:
            return False

    def insert_neighbour(self, corner, edge):
        self.corner_neighbours[self.neighbours_placed][0] = corner
        self.corner_neighbours[self.neighbours_placed][1] = edge
        self.neighbours_placed += 1

    def insert_adjacent_tile(self, tile):
        self.adjacent_tiles[self.adjacent_tiles_placed] = tile
        self.adjacent_tiles_placed += 1