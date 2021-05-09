from game.enums import Terrain, Resource

class Tile(object):
    def __init__(self, terrain: Terrain, value: int, id: int = None):
        self.terrain = terrain
        self.resource = Resource(terrain)
        self.value = value
        if value == 7:
            self.likelihood = None
        elif value == 6 or value == 8:
            self.likelihood = 5
        elif value == 5 or value == 9:
            self.likelihood = 4
        elif value == 4 or value == 10:
            self.likelihood = 3
        elif value == 3 or value == 11:
            self.likelihood = 2
        else:
            self.likelihood = 1
        self.id = id
        self.contains_robber = False

        self.corners = {
            "T": None,
            "TL": None,
            "BL": None,
            "B": None,
            "BR": None,
            "TR": None
        }
        self.edges = {
            "BL": None,
            "BR": None,
            "L": None,
            "R": None,
            "TL": None,
            "TR": None
        }

    def __eq__(self, other):
        if self.id == other.id:
            return True
        else:
            return False