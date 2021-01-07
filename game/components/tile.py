from game.enums import Terrain, Resource

class Tile(object):
    def __init__(self, terrain: Terrain, value: int, id: int = None):
        self.terrain = terrain
        self.resource = Resource(terrain)
        self.value = value
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