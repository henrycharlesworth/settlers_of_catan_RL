import numpy as np
from collections import defaultdict, deque

from game.components.board import Board
from game.components.player import Player
from game.enums import PlayerId, Resource, BuildingType, DevelopmentCard

from ui.display import Display

class Game(object):
    def __init__(self, board_config = {}):
        self.board = Board(**board_config)
        self.players = {
            PlayerId.Blue: Player(PlayerId.Blue),
            PlayerId.Red: Player(PlayerId.Red),
            PlayerId.Orange: Player(PlayerId.Orange),
            PlayerId.White: Player(PlayerId.White)
        }
        self.display = Display(self)
        self.reset()

    def render(self):
        self.display.render()

    def reset(self):
        self.board.reset()
        for player_id in self.players:
            self.players[player_id].reset()
        self.players_go = [PlayerId.White, PlayerId.Blue, PlayerId.Orange, PlayerId.Red]
        np.random.shuffle(self.players_go)
        self.resource_bank = {
            Resource.Sheep: 19,
            Resource.Wheat: 19,
            Resource.Brick: 19,
            Resource.Ore: 19,
            Resource.Wood: 19
        }
        self.building_bank = {
            "settlements": {
                PlayerId.Blue: 5,
                PlayerId.White: 5,
                PlayerId.Orange: 5,
                PlayerId.Red: 5
            },
            "cities": {
                PlayerId.Blue: 4,
                PlayerId.White: 4,
                PlayerId.Orange: 4,
                PlayerId.Red: 4
            }
        }
        self.road_bank = {
            PlayerId.Blue: 15,
            PlayerId.Red: 15,
            PlayerId.Orange: 15,
            PlayerId.White: 15
        }
        self.development_cards = [DevelopmentCard.Knight] * 14 + [DevelopmentCard.VictoryPoint] * 5 + \
            [DevelopmentCard.YearOfPlenty] * 2 + [DevelopmentCard.RoadBuilding] * 2 + [DevelopmentCard.Monopoly] * 2
        np.random.shuffle(self.development_cards)
        self.development_cards_pile = deque(self.development_cards)
        self.longest_road = None
        self.largest_army = None

        self.begin_initial_round = True

    def roll_dice(self):
        die_1 = np.random.randint(1, 7)
        die_2 = np.random.randint(1, 7)

        roll_value = int(die_1 + die_2)

        tiles_hit = self.board.value_to_tiles[roll_value]

        resources_allocated = {
            resource : defaultdict(lambda: 0) for resource in [Resource.Wood, Resource.Ore, Resource.Brick, Resource.Wheat, Resource.Sheep]
         }

        for tile in tiles_hit:
            if tile.contains_robber:
                continue
            for corner in tile.corners:
                if corner.building is not None:
                    if corner.building.type == BuildingType.Settlement:
                        increment = 1
                    elif corner.building.type == BuildingType.City:
                        increment = 2
                    resources_allocated[tile.resource][corner.building.owner] += increment
                    resources_allocated[tile.resource]["total"] += increment

        #allocate resources if we have enough in bank
        for resource in resources_allocated.keys():
            if resources_allocated[resource]["total"] <= self.resource_bank[resource]:
                for player in [PlayerId.Blue, PlayerId.Orange, PlayerId.White, PlayerId.Red]:
                    self.players[player].resources[resource] += resources_allocated[resource][player]
                    self.resource_bank[resource] -= resources_allocated[resource][player]

        return roll_value

    def can_buy_development_card(self, player):
        if player.resources[Resource.Wheat] > 0 and player.resources[Resource.Sheep] > 0 and \
                player.resources[Resource.Ore] > 0:
            return True
        else:
            return False

    def buy_development_card(self, player):
        player.resources[Resource.Wheat] -= 1
        player.resources[Resource.Sheep] -= 1
        player.resources[Resource.Ore] -= 1
        self.resource_bank[Resource.Wheat] += 1
        self.resource_bank[Resource.Sheep] += 1
        self.resource_bank[Resource.Ore] += 1

        card = self.development_cards_pile.pop()
        player.hidden_cards.append(card)