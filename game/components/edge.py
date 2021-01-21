from game.components.corner import Corner
from game.enums import PlayerId

class Edge(object):
    def __init__(self, id):
        self.id = id
        self.corner_1 = None
        self.corner_2 = None

        self.road = None
        self.harbour = None

    def __eq__(self, other):
        if self.id == other.id:
            return True
        else:
            return False

    def insert_corners(self, corner_1: Corner, corner_2: Corner):
        self.corner_1 = corner_1
        self.corner_2 = corner_2

    def can_place_road(self, player: PlayerId):
        if self.road is not None:
            return False
        else:
            if (self.corner_1.building is not None and self.corner_1.building.owner == player) or \
                    (self.corner_2.building is not None and self.corner_2.building.owner == player):
                return True
            else:
                for corner in [self.corner_1, self.corner_2]:
                    for next_corner in corner.corner_neighbours:
                        if next_corner[1] is not None:
                            if next_corner[1].road == player:
                                return True
        return False

    def insert_road(self, player: PlayerId):
        self.road = player