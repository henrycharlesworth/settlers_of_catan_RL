import copy

from game.enums import PlayerId, Resource

class Player(object):
    def __init__(self, id: PlayerId):
        self.id = id
        self.reset()

    def reset(self, player_order):
        self.player_order = player_order
        self.player_lookup = {}
        self.inverse_player_lookup = {}
        for i in range(len(player_order)):
            if player_order[i] == self.id:
                p_ind = i
        for i, label in enumerate(["next", "next_next", "next_next_next"]):
            ind = (p_ind + 1 + i) % 4
            self.player_lookup[self.player_order[ind]] = label
            self.inverse_player_lookup[label] = self.player_order[ind]

        self.buildings = []
        self.roads = []
        self.resources = {
            Resource.Brick: 5,
            Resource.Wood: 5,
            Resource.Wheat: 5,
            Resource.Ore: 5,
            Resource.Sheep: 5
        }
        self.visible_resources = {
            Resource.Brick: self.resources[Resource.Brick],
            Resource.Wood: self.resources[Resource.Wood],
            Resource.Wheat: self.resources[Resource.Wheat],
            Resource.Sheep: self.resources[Resource.Sheep],
            Resource.Ore: self.resources[Resource.Ore]
        }
        self.opponent_max_res = {
            "next": copy.copy(self.visible_resources),
            "next_next": copy.copy(self.visible_resources),
            "next_next_next": copy.copy(self.visible_resources)
        }
        self.opponent_min_res = copy.copy(self.opponent_max_res)
        self.harbours = {}
        self.longest_road = 0
        self.hidden_cards = []
        self.visible_cards = []
        self.victory_points = 0