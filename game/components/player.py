from game.enums import PlayerId, Resource

class Player(object):
    def __init__(self, id: PlayerId):
        self.id = id
        self.reset()

    def reset(self):
        self.buildings = []
        self.roads = []
        self.resources = {
            Resource.Brick: 0,
            Resource.Wood: 0,
            Resource.Wheat: 0,
            Resource.Ore: 0,
            Resource.Sheep: 0
        }
        self.harbours = {}
        self.longest_road = 0
        self.hidden_cards = []
        self.visible_cards = []
        self.victory_points = 0