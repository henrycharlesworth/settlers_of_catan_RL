import numpy as np
from collections import defaultdict, deque

from game.components.board import Board
from game.components.player import Player
from game.enums import PlayerId, Resource, BuildingType, DevelopmentCard, ActionTypes

from ui.display import Display

class Game(object):
    def __init__(self, board_config = {}, interactive=False):
        self.board = Board(**board_config)
        self.players = {
            PlayerId.Blue: Player(PlayerId.Blue),
            PlayerId.Red: Player(PlayerId.Red),
            PlayerId.Orange: Player(PlayerId.Orange),
            PlayerId.White: Player(PlayerId.White)
        }
        self.reset()
        self.display = Display(self, interactive=interactive)

    def render(self):
        self.display.render()

    def reset(self):
        self.board.reset()
        for player_id in self.players:
            self.players[player_id].reset()
        self.player_order = [PlayerId.White, PlayerId.Blue, PlayerId.Orange, PlayerId.Red]
        np.random.shuffle(self.player_order)
        self.players_go = self.player_order[0]
        self.player_order_id = 0
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

        self.initial_placement_phase = True
        self.initial_settlements_placed = {
            PlayerId.Blue: 0,
            PlayerId.Red: 0,
            PlayerId.Orange: 0,
            PlayerId.White: 0
        }
        self.initial_roads_placed = {
            PlayerId.Blue: 0,
            PlayerId.Red: 0,
            PlayerId.Orange: 0,
            PlayerId.White: 0
        }
        self.dice_rolled_this_turn = False
        self.played_development_card_this_turn = False
        self.can_move_robber = False

    def roll_dice(self):
        self.die_1 = np.random.randint(1, 7)
        self.die_2 = np.random.randint(1, 7)

        roll_value = int(self.die_1 + self.die_2)

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

        #allocate resources if we have enough in bank for all.
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

    def can_buy_settlement(self, player):
        if self.initial_placement_phase:
            return True
        if player.resources[Resource.Wheat] > 0 and player.resources[Resource.Wood] > 0 and \
                player.resources[Resource.Brick] and player.resources[Resource.Sheep] > 0:
            return True
        else:
            return False

    def build_settlement(self, player, corner):
        if self.initial_placement_phase == False:
            player.resources[Resource.Wheat] -= 1
            player.resources[Resource.Sheep] -= 1
            player.resources[Resource.Wood] -= 1
            player.resources[Resource.Brick] -= 1
        self.board.insert_settlement(player, corner, initial_placement=self.initial_placement_phase)
        player.victory_points += 1

    def can_buy_road(self, player):
        if self.initial_placement_phase:
            return True
        if player.resources[Resource.Wood] > 0 and player.resources[Resource.Brick] > 0:
            return True
        else:
            return False

    def build_road(self, player, edge):
        if self.initial_placement_phase == False:
            player.resources[Resource.Wood] -= 1
            player.resources[Resource.Brick] -= 1
        self.board.insert_road(player.id, edge)

    def can_buy_city(self, player):
        if player.resources[Resource.Wheat] > 1 and player.resources[Resource.Ore] > 2:
            return True
        return False

    def build_city(self, player, corner):
        player.resources[Resource.Wheat] -= 2
        player.resources[Resource.Ore] -= 3
        self.board.insert_city(player.id, corner)
        player.victory_points += 1

    def update_players_go(self, left=False):
        if left:
            self.player_order_id -= 1
            if self.player_order_id < 0:
                self.player_order_id = 3
        else:
            self.player_order_id += 1
            if self.player_order_id > 3:
                self.player_order_id = 0
        self.players_go = self.player_order[self.player_order_id]

    def validate_action(self, action):
        player = self.players[self.players_go]
        if action["type"] == ActionTypes.PlaceSettlement:
            if self.dice_rolled_this_turn == False and self.initial_placement_phase==False:
                return False
            if self.can_buy_settlement(player):
                corner = action["corner"]
                if self.board.corners[corner].can_place_settlement(player.id, initial_placement=self.initial_placement_phase):
                    if self.initial_placement_phase:
                        if self.initial_settlements_placed[self.players_go] == 0 or \
                                (self.initial_settlements_placed[self.players_go] == 1 and self.initial_roads_placed[self.players_go] == 1):
                            return True
                        return False
                    return True
            return False
        elif action["type"] == ActionTypes.PlaceRoad:
            if self.dice_rolled_this_turn == False and self.initial_placement_phase==False:
                return False
            if self.can_buy_road(player):
                edge = action["edge"]
                if self.board.edges[edge].can_place_road(player.id):
                    if self.initial_placement_phase:
                        if (self.initial_settlements_placed[self.players_go] == 1 and self.initial_roads_placed[self.players_go] == 0) or \
                                (self.initial_settlements_placed[self.players_go] == 2 and self.initial_roads_placed[self.players_go] == 1):
                            return True
                        return False
                    return True
            return False
        elif action["type"] == ActionTypes.UpgradeToCity:
            if self.dice_rolled_this_turn == False or self.initial_placement_phase:
                return False
            if self.can_buy_city(player):
                corner = self.board.corners[action["corner"]]
                if corner.building is not None and corner.building.type == BuildingType.Settlement:
                    if corner.building.owner == player.id:
                        return True
            return False
        elif action["type"] == ActionTypes.BuyDevelopmentCard:
            if self.dice_rolled_this_turn == False or self.initial_placement_phase:
                return False
            if self.can_buy_development_card(player):
                if len(self.development_cards_pile) > 0:
                    return True
            return False
        elif action["type"] == ActionTypes.PlayDevelopmentCard:
            if self.played_development_card_this_turn or self.initial_placement_phase:
                return False
            if action["card"] in player.hidden_cards:
                return True
            return False
        elif action["type"] == ActionTypes.HarbourTrade:
            if self.dice_rolled_this_turn == False or self.initial_placement_phase:
                return False
            harbour_resource = action["harbour_resource"]
            target_resource = action["desired_resource"]
            trading_resource = action["trading_resource"]
            harbour = player.harbours.get(harbour_resource, None)
            if harbour is not None:
                trade_ratio = harbour.exchange_value
                if player.resources[trading_resource] >= trade_ratio:
                    if self.resource_bank[target_resource] > 0:
                        return True
            return False
        elif action["type"] == ActionTypes.ProposeTrade:
            """allow for a target player, or for trade to be open to all"""
            pass
        elif action["type"] == ActionTypes.CounterOffer:
            pass
        elif action["type"] == ActionTypes.AcceptOffer:
            pass
        elif action["type"] == ActionTypes.MoveRobber:
            if self.can_move_robber:
                return True
            return False
        elif action["type"] == ActionTypes.RollDice:
            if self.dice_rolled_this_turn or self.initial_placement_phase:
                return False
            return True
        elif action["type"] == ActionTypes.EndTurn:
            if self.dice_rolled_this_turn == False or self.initial_placement_phase:
                return False
            return True

    def apply_action(self, action):
        """assuming validated action"""
        player = self.players[self.players_go]
        if action["type"] == ActionTypes.PlaceSettlement:
            corner = self.board.corners[action["corner"]]
            self.build_settlement(player, corner)
            if self.initial_placement_phase:
                self.initial_settlements_placed[player.id] += 1
                if self.initial_settlements_placed[player.id] == 2:
                    for tile in corner.adjacent_tiles:
                        if tile is None or tile.resource == Resource.Empty:
                            continue
                        player.resources[tile.resource] += 1
        elif action["type"] == ActionTypes.PlaceRoad:
            edge = self.board.edges[action["edge"]]
            self.build_road(player, edge)
            if self.initial_placement_phase:
                self.initial_roads_placed[player.id] += 1
                first_settlements_placed = 0
                second_settlements_placed = 0
                for player_id, count in self.initial_settlements_placed.items():
                    if count == 1:
                        first_settlements_placed += 1
                    elif count == 2:
                        first_settlements_placed += 1
                        second_settlements_placed += 1
                if first_settlements_placed < 4:
                    self.update_players_go()
                elif first_settlements_placed == 4 and second_settlements_placed == 0:
                    pass
                elif first_settlements_placed == 4 and second_settlements_placed < 4:
                    self.update_players_go(left=True)
                else:
                    self.initial_placement_phase = False
                    pass
