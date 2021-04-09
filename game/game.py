import numpy as np
import random
import copy
from collections import defaultdict, deque
from itertools import chain

from game.components.board import Board
from game.components.player import Player
from game.enums import PlayerId, Resource, BuildingType, DevelopmentCard, ActionTypes
from game.utils import DFS

from ui.display import Display

class Game(object):
    def __init__(self, board_config = {}, interactive=False, debug_mode=False):
        self.board = Board(**board_config)
        self.players = {
            PlayerId.Blue: Player(PlayerId.Blue),
            PlayerId.Red: Player(PlayerId.Red),
            PlayerId.Orange: Player(PlayerId.Orange),
            PlayerId.White: Player(PlayerId.White)
        }
        self.reset()
        self.interactive = interactive
        self.debug_mode = debug_mode
        if interactive:
            self.display = Display(self, interactive=interactive, debug_mode=debug_mode)
        else:
            self.display = None

    def render(self):
        if self.display is None:
            self.display = Display(self, interactive=self.interactive, debug_mode=self.debug_mode)

        self.display.render()

    def reset(self):
        self.board.reset()
        self.player_order = [PlayerId.White, PlayerId.Blue, PlayerId.Orange, PlayerId.Red]
        np.random.shuffle(self.player_order)
        for player_id in self.players:
            self.players[player_id].reset(self.player_order)

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
        self.initial_second_settlement_corners = {
            PlayerId.Blue: None,
            PlayerId.Red: None,
            PlayerId.Orange: None,
            PlayerId.White: None
        }
        self.dice_rolled_this_turn = False
        self.played_development_card_this_turn = False
        self.must_use_development_card_ability = False
        self.must_respond_to_trade = False
        self.proposed_trade = None
        self.road_building_active = [False, 0] #active, num roads placed
        self.can_move_robber = False
        self.just_moved_robber = False
        self.die_1 = None
        self.die_2 = None
        self.trades_proposed_this_turn = 0
        self.actions_this_turn = 0
        self.turn = 0
        self.development_cards_bought_this_turn = []
        self.longest_road = None
        self.largest_army = None
        self.colours = {
            PlayerId.White: (255, 255, 255),
            PlayerId.Red: (255, 0, 0),
            PlayerId.Blue: (0, 0, 255),
            PlayerId.Orange: (255, 153, 51)
        }
        self.resource_text = {
            Resource.Wood: "wood",
            Resource.Brick: "brick",
            Resource.Wheat: "wheat",
            Resource.Sheep: "sheep",
            Resource.Ore: "ore"
        }

    def roll_dice(self):
        self.die_1 = np.random.randint(1, 7)
        self.die_2 = np.random.randint(1, 7)

        roll_value = int(self.die_1 + self.die_2)

        if roll_value == 7:
            return roll_value

        tiles_hit = self.board.value_to_tiles[roll_value]

        resources_allocated = {
            resource : defaultdict(lambda: 0) for resource in [Resource.Wood, Resource.Ore, Resource.Brick, Resource.Wheat, Resource.Sheep]
         }

        for tile in tiles_hit:
            if tile.contains_robber:
                continue
            for corner_key, corner in tile.corners.items():
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
                    self.update_player_resource_estimates({resource: resources_allocated[resource][player]}, player)

        return roll_value

    def can_buy_development_card(self, player):
        if player.resources[Resource.Wheat] > 0 and player.resources[Resource.Sheep] > 0 and \
                player.resources[Resource.Ore] > 0:
            return True
        else:
            return False

    def can_buy_settlement(self, player):
        if self.initial_placement_phase:
            return True
        if self.building_bank["settlements"][player.id] > 0:
            if player.resources[Resource.Wheat] > 0 and player.resources[Resource.Wood] > 0 and \
                    player.resources[Resource.Brick] > 0 and player.resources[Resource.Sheep] > 0:
                return True
        return False

    def build_settlement(self, player, corner):
        if self.initial_placement_phase == False:
            player.resources[Resource.Wheat] -= 1
            player.resources[Resource.Sheep] -= 1
            player.resources[Resource.Wood] -= 1
            player.resources[Resource.Brick] -= 1
            player.visible_resources[Resource.Wheat] = max(player.visible_resources[Resource.Wheat] - 1, 0)
            player.visible_resources[Resource.Sheep] = max(player.visible_resources[Resource.Sheep] - 1, 0)
            player.visible_resources[Resource.Wood] = max(player.visible_resources[Resource.Wood] - 1, 0)
            player.visible_resources[Resource.Brick] = max(player.visible_resources[Resource.Brick] - 1, 0)
            self.resource_bank[Resource.Wheat] += 1
            self.resource_bank[Resource.Sheep] += 1
            self.resource_bank[Resource.Wood] += 1
            self.resource_bank[Resource.Brick] += 1
        self.board.insert_settlement(player, corner, initial_placement=self.initial_placement_phase)
        self.building_bank["settlements"][player.id] -= 1
        player.victory_points += 1

    def can_buy_road(self, player):
        if self.initial_placement_phase:
            return True
        if player.resources[Resource.Wood] > 0 and player.resources[Resource.Brick] > 0:
            return True
        else:
            return False

    def build_road(self, player, edge, road_building=False):
        if self.initial_placement_phase == False:
            if road_building == False:
                player.resources[Resource.Wood] -= 1
                player.resources[Resource.Brick] -= 1
                player.visible_resources[Resource.Wood] = max(player.visible_resources[Resource.Wood] - 1, 0)
                player.visible_resources[Resource.Brick] = max(player.visible_resources[Resource.Brick] - 1, 0)
                self.resource_bank[Resource.Wood] += 1
                self.resource_bank[Resource.Brick] += 1
        self.board.insert_road(player.id, edge)

    def can_buy_city(self, player):
        if self.building_bank["cities"][player.id] > 0:
            if player.resources[Resource.Wheat] > 1 and player.resources[Resource.Ore] > 2:
                return True
        return False

    def build_city(self, player, corner):
        player.resources[Resource.Wheat] -= 2
        player.resources[Resource.Ore] -= 3
        player.visible_resources[Resource.Wheat] = max(player.visible_resources[Resource.Wheat] - 2, 0)
        player.visible_resources[Resource.Ore] = max(player.visible_resources[Resource.Ore] - 3, 0)
        self.resource_bank[Resource.Wheat] += 2
        self.resource_bank[Resource.Ore] += 3
        self.board.insert_city(player.id, corner)
        player.victory_points += 1
        self.building_bank["cities"][player.id] -= 1
        self.building_bank["settlements"][player.id] += 1

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
            if self.must_respond_to_trade:
                return False, "Must respond to proposed trade."
            if (self.dice_rolled_this_turn == False and self.initial_placement_phase==False):
                return False, "You need to roll the dice first!"
            elif self.must_use_development_card_ability:
                return False, "You need to play out your development card ability first."
            elif self.just_moved_robber:
                return False, "You've just moved the robber. Choose a player to steal from first."
            if self.can_buy_settlement(player):
                corner = action["corner"]
                if self.board.corners[corner].can_place_settlement(player.id, initial_placement=self.initial_placement_phase):
                    if self.initial_placement_phase:
                        if self.initial_settlements_placed[self.players_go] == 0 or \
                                (self.initial_settlements_placed[self.players_go] == 1 and self.initial_roads_placed[self.players_go] == 1):
                            return True, None
                        return False, "You cannot place a settlement here!"
                    return True, None
            return False, "You cannot afford a settlement!"
        elif action["type"] == ActionTypes.PlaceRoad:
            if self.road_building_active[0]:
                edge = action["edge"]
                if edge is None: #dummy edge
                    return True, None
                if self.board.edges[edge].can_place_road(player.id):
                    return True, None
                else:
                    return False, "Cannot place a road here."
            if self.must_respond_to_trade:
                return False, "Must respond to proposed trade."
            if (self.dice_rolled_this_turn == False and self.initial_placement_phase==False):
                return False, "You need to roll the dice first!"
            elif self.must_use_development_card_ability:
                return False, "You need to play out your development card ability first."
            elif self.just_moved_robber:
                return False, "You've just moved the robber. Choose a player to steal from first."
            if self.can_buy_road(player):
                edge = action["edge"]
                if self.board.edges[edge].can_place_road(player.id):
                    if self.initial_placement_phase:
                        if (self.initial_settlements_placed[self.players_go] == 1 and self.initial_roads_placed[self.players_go] == 0):
                            return True, None
                        elif (self.initial_settlements_placed[self.players_go] == 2 and self.initial_roads_placed[self.players_go] == 1):
                            if self.board.edges[edge].can_place_road(player.id, after_second_settlement=True,
                                                second_settlement=self.initial_second_settlement_corners[player.id]):
                                return True, None
                            else:
                                return False, "Must place second road next to second settlement."
                        return False, "You cannot place a road here!"
                    return True, None
                return False, "You cannot place a road here!"
            else:
                return False, "You cannot afford a road!"
        elif action["type"] == ActionTypes.UpgradeToCity:
            if self.must_respond_to_trade:
                return False, "Must respond to proposed trade."
            if self.initial_placement_phase:
                return False, "Still in initial placement phase!"
            elif self.dice_rolled_this_turn == False:
                return False, "You need to roll the dice first!"
            elif self.must_use_development_card_ability:
                return False, "You need to play out your development card ability first."
            elif  self.just_moved_robber:
                return False, "You've just moved the robber. Choose a player to steal from first."
            if self.can_buy_city(player):
                corner = self.board.corners[action["corner"]]
                if corner.building is not None and corner.building.type == BuildingType.Settlement:
                    if corner.building.owner == player.id:
                        return True, None
                else:
                    return False, "This cannot be upgraded to a city!"
            return False, "You cannot afford to upgrade to a city!"
        elif action["type"] == ActionTypes.BuyDevelopmentCard:
            if self.must_respond_to_trade:
                return False, "Must respond to proposed trade."
            if self.initial_placement_phase:
                return False, "Still in initial placement phase. Cannot buy a development card"
            elif self.dice_rolled_this_turn == False:
                return False, "You need to roll the dice first!"
            elif self.must_use_development_card_ability:
                return False, "You need to play out your development card ability first."
            elif self.just_moved_robber:
                return False, "You've just moved the robber. Choose a player to steal from first."
            if self.can_buy_development_card(player):
                if len(self.development_cards_pile) > 0:
                    return True, None
                else:
                    return False, "No development cards left to buy."
            return False, "Cannot afford a development card!"
        elif action["type"] == ActionTypes.PlayDevelopmentCard:
            if self.must_respond_to_trade:
                return False, "Must respond to proposed trade."
            if self.played_development_card_this_turn:
                return False, "You have already played one development card this turn."
            elif self.initial_placement_phase:
                return False, "Still in initial placement phase!"
            elif self.just_moved_robber:
                return False, "You've just moved the robber. Choose a player to steal from first."
            if action["card"] in player.hidden_cards:
                if player.hidden_cards.count(action["card"]) > 0:
                    if player.hidden_cards.count(action["card"]) == self.development_cards_bought_this_turn.count(action["card"]):
                        return False, "Cannot play a card the same turn as it was bought."
                if action["card"] == DevelopmentCard.YearOfPlenty:
                    return True, None
                elif action["card"] == DevelopmentCard.Monopoly:
                    if "resource" not in action:
                        return False, "Must select which resource to steal."
                    else:
                        return True, None
                return True, None
            return False, "You cannot afford a development card!"
        elif action["type"] == ActionTypes.ExchangeResource:
            if self.must_respond_to_trade:
                return False, "Must respond to proposed trade."
            if self.initial_placement_phase:
                return False, "Still in initial placement phase!"
            elif self.dice_rolled_this_turn == False:
                return False, "You haven't rolled the dice yet!"
            elif self.must_use_development_card_ability:
                return False, "You need to play out your development card ability first."
            elif self.just_moved_robber:
                return False, "You've just moved the robber. Choose a player to steal from first."

            target_resource = action["desired_resource"]
            trading_resource = action["trading_resource"]

            if action.get("harbour", -1) is not -1:
                res = action["harbour"]
                harbour = player.harbours.get(res, -1)
                if isinstance(harbour, int) and harbour == -1:
                    return False, "You do not have access to this harbour."
            trade_ratio = action["exchange_rate"]
            if player.resources[trading_resource] >= trade_ratio:
                if self.resource_bank[target_resource] > 0:
                    return True, None
                else:
                    return False, "Not enough of desired resource left in the bank."
            else:
                return False, "You do not have enough resources for this exchange."
        elif action["type"] == ActionTypes.ProposeTrade:
            if self.must_respond_to_trade:
                return False, "Must respond to proposed trade."
            if self.initial_placement_phase:
                return False, "Still in initial placement phase!"
            elif self.dice_rolled_this_turn == False:
                return False, "You cannot trade before rolling the dice."
            elif self.must_use_development_card_ability:
                return False, "You need to play out your development card ability first."
            elif self.just_moved_robber:
                return False, "You've just moved the robber. Choose a player to steal from first."
            proposing_player = action["player_proposing"]
            other_player = action["target_player"]
            if other_player == proposing_player:
                return False, "Cannot trade with yourself."
            proposed_res = action["player_proposing_res"]
            for res in [Resource.Wood, Resource.Brick, Resource.Sheep, Resource.Wheat, Resource.Ore]:
                prop_count = proposed_res.count(res)
                if player.resources[res] >= prop_count:
                    continue
                else:
                    return False, "You do not have the proposed resources!"
            return True, None
        elif action["type"] == ActionTypes.RespondToOffer:
            if self.must_respond_to_trade:
                target_player = self.players[self.proposed_trade["target_player"]]
                if action["response"] == "reject":
                    return True, None
                else:
                    proposed_res = self.proposed_trade["target_player_res"]
                    for res in [Resource.Wood, Resource.Brick, Resource.Sheep, Resource.Wheat, Resource.Ore]:
                        prop_count = proposed_res.count(res)
                        if target_player.resources[res] >= prop_count:
                            continue
                        else:
                            return False, "You do not have enough resources to accept this trade."
                    return True, None
            else:
                return False, "No trade to respond to."
        elif action["type"] == ActionTypes.MoveRobber:
            if self.must_respond_to_trade:
                return False, "Must respond to proposed trade."
            if self.must_use_development_card_ability:
                return False, "You need to play out your development card ability first."
            if self.can_move_robber:
                return True, None
            return False
        elif action["type"] == ActionTypes.RollDice:
            if self.must_respond_to_trade:
                return False, "Must respond to proposed trade."
            if self.initial_placement_phase:
                return False, "Still in initial placement phase!"
            elif self.dice_rolled_this_turn:
                return False, "You have already rolled the dice this turn!"
            elif self.just_moved_robber:
                return False, "You've just moved the robber. Choose a player to steal from first."
            return True, None
        elif action["type"] == ActionTypes.EndTurn:
            if self.must_respond_to_trade:
                return False, "Must respond to proposed trade."
            if self.initial_placement_phase:
                return False, "Still in initial placement phase!"
            elif self.dice_rolled_this_turn == False:
                return False, "You cannot end your turn before rolling the dice!"
            elif self.must_use_development_card_ability:
                return False, "You need to play out your development card ability first."
            elif self.just_moved_robber:
                return False, "You've just moved the robber. Choose a player to steal from first."
            return True, None
        elif action["type"] == ActionTypes.StealResource:
            if self.must_respond_to_trade:
                return False, "Must respond to proposed trade."
            if self.just_moved_robber:
                player = action["target"]
                robber_tile = self.board.robber_tile
                for key, val in robber_tile.corners.items():
                    if val is not None and val.building is not None:
                        if val.building.owner == player:
                            return True, None
                return False, "Cannot steal from a player who doesn't have a building on tile with robber."
            else:
                return False, "You cannot steal a resource when you haven't moved the robber."

    def apply_action(self, action, return_message=True):
        """assuming validated action"""
        player = self.players[self.players_go]
        if action["type"] == ActionTypes.PlaceSettlement:
            corner = self.board.corners[action["corner"]]
            self.build_settlement(player, corner)
            if self.initial_placement_phase:
                self.initial_settlements_placed[player.id] += 1
                if self.initial_settlements_placed[player.id] == 2:
                    tile_res = defaultdict(lambda: 0)
                    for tile in corner.adjacent_tiles:
                        if tile is None or tile.resource == Resource.Empty:
                            continue
                        player.resources[tile.resource] += 1
                        player.visible_resources[tile.resource] += 1
                        tile_res[tile.resource] += 1
                        self.resource_bank[tile.resource] -= 1
                    self.update_player_resource_estimates(tile_res, player.id)
                    self.initial_second_settlement_corners[player.id] = action["corner"]
            else:
                building_res = {
                    Resource.Brick: -1, Resource.Wood: -1, Resource.Wheat: -1, Resource.Sheep: -1
                }
                self.update_player_resource_estimates(building_res, player.id)
                #settlements can block longest road, so need to check.
                if self.longest_road is not None:
                    self.update_longest_road(self.longest_road["player"])
            if return_message:
                message = {"player_id": player.id, "text": "Placed a settlement."}
        elif action["type"] == ActionTypes.PlaceRoad:
            if action["edge"] is None:
                edge = None
            else:
                edge = self.board.edges[action["edge"]]
            placing_final_init_road = False
            if edge is None:
                message = {"player_id": player.id, "text": "Placed dummy road (nothing happened)."}
            else:
                self.build_road(player, edge, road_building=self.road_building_active[0])
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
                        placing_final_init_road = True
            self.update_longest_road(player.id)
            if self.road_building_active[0]:
                self.road_building_active[1] += 1
                if self.road_building_active[1] >= 2:
                    self.road_building_active = [False, 0]
                    self.must_use_development_card_ability = False
            else:
                if self.initial_placement_phase == False and placing_final_init_road == False:
                    self.update_player_resource_estimates(
                        {Resource.Brick: -1, Resource.Wood: -1}, player.id
                    )
            if return_message and action["edge"] is not None:
                message = {"player_id": player.id, "text": "Placed a road."}
        elif action["type"] == ActionTypes.UpgradeToCity:
            self.build_city(player, self.board.corners[action["corner"]])
            if return_message:
                message = {"player_id": player.id, "text": "Upgraded their settlement into a city."}
            self.update_player_resource_estimates(
                {Resource.Ore: -3, Resource.Wheat: -2}, player.id
            )
        elif action["type"] == ActionTypes.RollDice:
            roll_value = self.roll_dice()
            self.dice_rolled_this_turn = True
            if roll_value == 7:
                self.can_move_robber = True
            if return_message:
                message = {"player_id": player.id, "text": "Rolled the dice (" + str(roll_value) + ")."}
        elif action["type"] == ActionTypes.EndTurn:
            self.can_move_robber = False
            self.dice_rolled_this_turn = False
            self.played_development_card_this_turn = False
            self.update_players_go()
            self.turn += 1
            self.development_cards_bought_this_turn = []
            self.trades_proposed_this_turn = 0
            self.actions_this_turn = 0
            if return_message:
                message = {"player_id": player.id, "text": "Ended their turn."}
        elif action["type"] == ActionTypes.MoveRobber:
            self.board.move_robber(self.board.tiles[action["tile"]])
            self.can_move_robber = False
            tile_corner_values = self.board.tiles[action["tile"]].corners.values()
            player_to_steal_from = False
            for value in tile_corner_values:
                if value.building is not None and value.building.owner != player.id:
                    player_to_steal_from = True
            if player_to_steal_from:
                self.just_moved_robber = True
            if return_message:
                message = {"player_id": player.id, "text": "Moved the robber."}
        elif action["type"] == ActionTypes.StealResource:
            player_id = action["target"]
            resources_to_steal = []
            for res in [Resource.Brick, Resource.Wheat, Resource.Wood, Resource.Sheep, Resource.Ore]:
                resources_to_steal += [res] * self.players[player_id].resources[res]
            if len(resources_to_steal) == 0:
                pass
            else:
                resource_to_steal = random.choice(resources_to_steal)
                player.resources[resource_to_steal] += 1
                self.players[player_id].resources[resource_to_steal] -= 1
                for res in [Resource.Brick, Resource.Wheat, Resource.Wood, Resource.Sheep, Resource.Ore]:
                    self.players[player_id].visible_resources[res] = max(self.players[player_id].visible_resources[res] - 1, 0)
                self.update_player_resource_estimates({resource_to_steal: -1}, player_id, player.id)
            self.just_moved_robber = False
            if return_message:
                message = {"player_id": player.id}
                message["text"] = "Stole a resource from {style}{color " + str(self.colours[action["target"]]) + "}Player"
        elif action["type"] == ActionTypes.PlayDevelopmentCard:
            player.hidden_cards.remove(action["card"])
            player.visible_cards.append(action["card"])
            self.played_development_card_this_turn = True
            if action["card"] == DevelopmentCard.VictoryPoint:
                player.victory_points += 1
                card_type_text = "Victory Point card."
            elif action["card"] == DevelopmentCard.Knight:
                self.can_move_robber = True
                self.update_largest_army()
                card_type_text = "Knight card."
            elif action["card"] == DevelopmentCard.RoadBuilding:
                self.road_building_active[0] = True
                self.road_building_active[1] = 0
                self.must_use_development_card_ability = True
                card_type_text = "Road Building card."
            elif action["card"] == DevelopmentCard.Monopoly:
                resource = action["resource"]
                card_type_text = "Monopoly card, taking " + self.resource_text[resource] + "."
                num_res = {}
                for other_player in self.players.values():
                    if other_player.id == player.id:
                        continue
                    res_count = other_player.resources[resource]
                    other_player.resources[resource] = 0
                    other_player.visible_resources[resource] = 0
                    player.resources[resource] += res_count
                    player.visible_resources[resource] += res_count
                    num_res[other_player.id] = res_count
                self.update_resource_estimates_monopoly(player.id, resource, num_res)
            elif action["card"] == DevelopmentCard.YearOfPlenty:
                for resource in [action["resource_1"], action["resource_2"]]:
                    if self.resource_bank[resource] > 0:
                        self.resource_bank[resource] -= 1
                        player.resources[resource] += 1
                        player.visible_resources[resource] += 1
                card_type_text = "Year of Plenty card, collecting {} and {}.".format(self.resource_text[action["resource_1"]],
                                                                                     self.resource_text[action["resource_2"]])
            if return_message:
                message = {"player_id": player.id, "text": "Played a "+card_type_text}
        elif action["type"] == ActionTypes.BuyDevelopmentCard:
            player.resources[Resource.Sheep] -= 1
            player.resources[Resource.Ore] -= 1
            player.resources[Resource.Wheat] -= 1
            player.visible_resources[Resource.Sheep] = max(player.visible_resources[Resource.Sheep] - 1, 0)
            player.visible_resources[Resource.Ore] = max(player.visible_resources[Resource.Ore] - 1, 0)
            player.visible_resources[Resource.Wheat] = max(player.visible_resources[Resource.Wheat] - 1, 0)
            self.resource_bank[Resource.Sheep] += 1
            self.resource_bank[Resource.Ore] += 1
            self.resource_bank[Resource.Wheat] += 1
            self.update_player_resource_estimates(
                {Resource.Sheep: -1, Resource.Ore: -1, Resource.Wheat: -1}, player.id
            )
            player.hidden_cards.append(self.development_cards_pile.pop())
            self.development_cards_bought_this_turn.append(copy.copy(player.hidden_cards[-1]))
            if return_message:
                message = {"player_id": player.id, "text": "Bought a development card."}
        elif action["type"] == ActionTypes.ExchangeResource:
            desired_resource = action["desired_resource"]
            trade_resource = action["trading_resource"]
            player.resources[desired_resource] += 1
            player.visible_resources[desired_resource] += 1
            player.resources[trade_resource] -= action["exchange_rate"]
            player.visible_resources[trade_resource] = max(player.visible_resources[trade_resource] - action["exchange_rate"], 0)
            self.resource_bank[trade_resource] += action["exchange_rate"]
            self.resource_bank[desired_resource] -= 1
            self.update_player_resource_estimates(
                {desired_resource: 1, trade_resource: -action["exchange_rate"]}, player.id
            )
            if return_message:
                message = {
                    "player_id": player.id,
                    "text": "Exchanged {} {} for {}".format(action["exchange_rate"],
                                                              self.resource_text[action["trading_resource"]],
                                                              self.resource_text[action["desired_resource"]])
                }
        elif action["type"] == ActionTypes.ProposeTrade:
            self.must_respond_to_trade = True
            self.proposed_trade = action.copy()
            if return_message:
                message = {"player_id": player.id}
                text = "Proposed a deal to {style}{color " + str(self.colours[action["target_player"]]) + \
                            "}Player, {style}{color " + str(self.colours[player.id]) + "}offering "
                for res in self.proposed_trade["player_proposing_res"]:
                    text += self.resource_text[res] + ", "
                text = text[:-2]
                text += " for "
                for res in self.proposed_trade["target_player_res"]:
                    text += self.resource_text[res] + ", "
                text = text[:-2] + "."
                message["text"] = text
                self.trades_proposed_this_turn += 1
        elif action["type"] == ActionTypes.RespondToOffer:
            if action["response"] == "accept":
                trade = self.proposed_trade
                p1 = trade["player_proposing"]
                p2 = trade["target_player"]
                res_p1 = defaultdict(lambda: 0)
                res_p2 = defaultdict(lambda: 0)
                for res in trade["player_proposing_res"]:
                    self.players[p1].resources[res] -= 1
                    self.players[p1].visible_resources[res] = max(self.players[p1].visible_resources[res] - 1, 0)
                    res_p1[res] -= 1
                    self.players[p2].resources[res] += 1
                    self.players[p2].visible_resources[res] += 1
                    res_p2[res] += 1
                for res in trade["target_player_res"]:
                    self.players[p1].resources[res] += 1
                    self.players[p1].visible_resources[res] += 1
                    res_p1[res] += 1
                    self.players[p2].resources[res] -= 1
                    self.players[p2].visible_resources[res] = max(self.players[p2].visible_resources[res] - 1, 0)
                    res_p2[res] -= 1
                self.update_player_resource_estimates(res_p1, p1)
                self.update_player_resource_estimates(res_p2, p2)
            self.must_respond_to_trade = False
            if return_message:
                message = {"player_id": self.proposed_trade["target_player"]}
                if action["response"] == "accept":
                    text = "Accepted the deal."
                else:
                    text = "Rejected the deal."
                message["text"] = text
                self.proposed_trade = None
            else:
                self.proposed_trade = None
        if action["type"] != ActionTypes.RespondToOffer and action["type"] != ActionTypes.EndTurn:
            self.actions_this_turn += 1

        if self.interactive == False and self.display is not None:
            self.display.update_game_log(message)
        if return_message:
            return message

    def update_largest_army(self):
        max_count = 0
        count_player = None
        for player_id in [PlayerId.Blue, PlayerId.White, PlayerId.Red, PlayerId.Orange]:
            knight_count = self.players[player_id].visible_cards.count(DevelopmentCard.Knight)
            if knight_count >= 3 and knight_count > max_count:
                max_count = knight_count
                count_player = player_id
        if count_player is not None:
            largest_army_update = {
                "player": count_player,
                "count": max_count
            }
            if self.largest_army is None:
                self.largest_army = largest_army_update
                self.players[count_player].victory_points += 2
            else:
                if self.largest_army["player"] == count_player:
                    self.largest_army = largest_army_update
                else:
                    if max_count > self.largest_army["count"]:
                        self.players[self.largest_army["player"]].victory_points -= 2
                        self.largest_army = largest_army_update
                        self.players[count_player].victory_points += 2

    def get_longest_path(self, player_id):
        player_edges = []
        for edge in self.board.edges:
            if edge.road is not None and edge.road == player_id:
                player_edges.append([edge.corner_1.id, edge.corner_2.id])

        G = defaultdict(list)
        for (s, t) in player_edges:
            if self.board.corners[s].building is not None and self.board.corners[s].building.owner != player_id:
                pass
            else:
                G[s].append(t)
            if self.board.corners[t].building is not None and self.board.corners[t].building.owner != player_id:
                pass
            else:
                G[t].append(s)

        all_paths = list(chain.from_iterable(DFS(G, n) for n in set(G)))
        max_path_len = max(len(p) - 1 for p in all_paths)
        return max_path_len

    def update_longest_road(self, player_id):
        max_path_len = self.get_longest_path(player_id)

        longest_road_update = {
            "player": player_id,
            "count": max_path_len
        }

        if self.longest_road is None:
            if max_path_len >= 5:
                self.longest_road = longest_road_update
                self.players[player_id].victory_points += 2
            return

        if self.longest_road["player"] == player_id:
            if self.longest_road["count"] > max_path_len:
                #longest road has been cut off, at least partially.
                max_p_len = max_path_len
                player = player_id
                tied_longest_road = False
                for other_pid in [PlayerId.White, PlayerId.Blue, PlayerId.Orange, PlayerId.Red]:
                    if other_pid == player_id:
                        continue
                    p_len = self.get_longest_path(other_pid)
                    if p_len == max_p_len:
                        tied_longest_road = True
                    elif p_len > max_p_len:
                        max_p_len = p_len
                        tied_longest_road = False
                        player = other_pid
                if max_p_len >= 5:
                    if tied_longest_road:
                        if player == player_id:
                            self.longest_road = longest_road_update
                        else:
                            self.longest_road = None
                            self.players[player_id].victory_points -= 2
                    else:
                        self.longest_road = {
                            "player": player,
                            "count": max_p_len
                        }
                        self.players[player].victory_points += 2
                        self.players[player_id].victory_points -= 2
                else:
                    self.longest_road = None
                    self.players[player_id].victory_points -= 2
            else:
                self.longest_road = longest_road_update
        else:
            if max_path_len > self.longest_road["count"]:
                self.players[self.longest_road["player"]].victory_points -= 2
                self.players[player_id].victory_points += 2
                self.longest_road = longest_road_update

    def update_player_resource_estimates(self, resources, player_updating, player_who_stole = None):
        total_resources = sum(self.players[player_updating].resources.values())
        if player_who_stole is not None:
            total_resources_stealer = sum(self.players[player_who_stole].resources.values())
        for player in [PlayerId.White, PlayerId.Red, PlayerId.Blue, PlayerId.Orange]:
            if player == player_updating:
                if player_who_stole is None:
                    continue
                stealing_player_label = self.players[player].player_lookup[player_who_stole]
                for res, num_res in resources.items():
                    #player knows what was stolen
                    self.players[player].opponent_max_res[stealing_player_label][res] -= num_res
                    self.players[player].opponent_min_res[stealing_player_label][res] -= num_res
            else:
                player_label = self.players[player].player_lookup[player_updating]
                if player_who_stole is None:
                    # visible exchange of resources
                    for res, num_res in resources.items():
                        curr_max_est = self.players[player].opponent_max_res[player_label][res]
                        curr_min_est = self.players[player].opponent_min_res[player_label][res]
                        self.players[player].opponent_max_res[player_label][res] = np.clip(curr_max_est + num_res,
                                                                                           0, total_resources)
                        self.players[player].opponent_min_res[player_label][res] = np.clip(curr_min_est + num_res,
                                                                                           0, total_resources)
                else:
                    if player == player_who_stole:
                        # knows exactly what was stolen
                        for res, num_res in resources.items():
                            curr_max_est = self.players[player].opponent_max_res[player_label][res]
                            curr_min_est = self.players[player].opponent_min_res[player_label][res]
                            self.players[player].opponent_max_res[player_label][res] = np.clip(curr_max_est + num_res,
                                                                                               0, total_resources)
                            self.players[player].opponent_min_res[player_label][res] = np.clip(curr_min_est + num_res,
                                                                                               0, total_resources)
                    else:
                        stealing_player_label = self.players[player].player_lookup[player_who_stole]
                        for res in [Resource.Wood, Resource.Brick, Resource.Wheat, Resource.Sheep, Resource.Ore]:
                            curr_max_player_est = self.players[player].opponent_max_res[player_label][res]
                            curr_min_player_est = self.players[player].opponent_min_res[player_label][res]
                            self.players[player].opponent_max_res[player_label][res] = np.clip(curr_max_player_est,
                                                                                               0, total_resources)
                            #any resource could have been stolen
                            self.players[player].opponent_min_res[player_label][res] = np.clip(curr_min_player_est - 1,
                                                                                               0, total_resources)
                            if curr_max_player_est > 0:
                                curr_max_steal_est = self.players[player].opponent_max_res[stealing_player_label][res]
                                curr_min_steal_est = self.players[player].opponent_min_res[stealing_player_label][res]
                                self.players[player].opponent_max_res[stealing_player_label][res] = np.clip(curr_max_steal_est + 1,
                                                                                                            0, total_resources_stealer)
                                self.players[player].opponent_min_res[stealing_player_label][res] = np.clip(curr_min_steal_est,
                                                                                                            0, total_resources_stealer)

    def update_resource_estimates_monopoly(self, player_id, res, num_res):
        """
        Forgot about this - easier to do it separately now.
        """
        total_res = 0
        for player in [PlayerId.White, PlayerId.Orange, PlayerId.Blue, PlayerId.Red]:
            if player == player_id:
                continue
            total_res += num_res[player]

        for player in [PlayerId.White, PlayerId.Orange, PlayerId.Blue, PlayerId.Red]:
            if player == player_id:
                #every player knows what they gained
                for o_player in [PlayerId.White, PlayerId.Orange, PlayerId.Blue, PlayerId.Red]:
                    if o_player == player:
                        continue
                    player_label = self.players[o_player].player_lookup[player]
                    self.players[o_player].opponent_min_res[player_label][res] += total_res
                    self.players[o_player].opponent_max_res[player_label][res] += total_res
            else:
                res_lost = num_res[player]
                player_total_res = 0
                for o_res in [Resource.Wood, Resource.Brick, Resource.Sheep, Resource.Wheat, Resource.Ore]:
                    player_total_res += self.players[player].resources[o_res]
                for o_player in [PlayerId.White, PlayerId.Orange, PlayerId.Blue, PlayerId.Red]:
                    if o_player == player:
                        continue
                    player_label = self.players[o_player].player_lookup[player]
                    for o_res in [Resource.Wood, Resource.Brick, Resource.Sheep, Resource.Wheat, Resource.Ore]:
                        curr_max_est = self.players[o_player].opponent_max_res[player_label][o_res]
                        curr_min_est = self.players[o_player].opponent_min_res[player_label][o_res]
                        if o_res == res:
                            curr_min_est -= res_lost
                            curr_max_est -= res_lost
                        self.players[o_player].opponent_max_res[player_label][o_res] = np.clip(curr_max_est, 0,
                                                                                               player_total_res)
                        self.players[o_player].opponent_min_res[player_label][o_res] = np.clip(curr_min_est, 0,
                                                                                               player_total_res)