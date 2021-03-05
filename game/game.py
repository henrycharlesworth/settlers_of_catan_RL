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
        self.must_use_development_card_ability = False
        self.must_respond_to_trade = False
        self.proposed_trade = None
        self.road_building_active = [False, 0] #active, num roads placed
        self.can_move_robber = False
        self.just_moved_robber = False
        self.die_1 = None
        self.die_2 = None
        self.max_trade_resources = 4
        self.turn = 0
        self.development_cards_bought_this_turn = []
        self.longest_road = None
        self.largest_army = None

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
        if self.building_bank["settlements"][player.id] > 0:
            if player.resources[Resource.Wheat] > 0 and player.resources[Resource.Wood] > 0 and \
                    player.resources[Resource.Brick] and player.resources[Resource.Sheep] > 0:
                return True
        return False

    def build_settlement(self, player, corner):
        if self.initial_placement_phase == False:
            player.resources[Resource.Wheat] -= 1
            player.resources[Resource.Sheep] -= 1
            player.resources[Resource.Wood] -= 1
            player.resources[Resource.Brick] -= 1
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
        self.resource_bank[Resource.Wheat] += 2
        self.resource_bank[Resource.Ore] += 3
        self.board.insert_city(player.id, corner)
        player.victory_points += 1
        self.building_bank["cities"][player.id] -= 1

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
                        if (self.initial_settlements_placed[self.players_go] == 1 and self.initial_roads_placed[self.players_go] == 0) or \
                                (self.initial_settlements_placed[self.players_go] == 2 and self.initial_roads_placed[self.players_go] == 1):
                            return True, None
                        return False, "You cannot place a road here!"
                    return True, None
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
                if player.hidden_cards.count(action["card"]) == 1:
                    if action["card"] in self.development_cards_bought_this_turn:
                        return False, "Cannot play a card the same turn as it was bought."
                if action["card"] == DevelopmentCard.YearOfPlenty:
                    if self.resource_bank[action["resource_1"]] > 0 and self.resource_bank[action["resource_2"]] > 0:
                        return True, None
                    else:
                        return False, "Not enough of these resources left in the bank."
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
                if action["response"] == "reject":
                    return True, None
                else:
                    proposed_res = self.proposed_trade["target_player_res"]
                    for res in [Resource.Wood, Resource.Brick, Resource.Sheep, Resource.Wheat, Resource.Ore]:
                        prop_count = proposed_res.count(res)
                        if player.resources[res] >= prop_count:
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
                    pass
            self.update_longest_road(player.id)
            if self.road_building_active[0]:
                self.road_building_active[1] += 1
                if self.road_building_active[1] >= 2:
                    self.road_building_active = [False, 0]
                    self.must_use_development_card_ability = False
        elif action["type"] == ActionTypes.UpgradeToCity:
            self.build_city(player, self.board.corners[action["corner"]])
        elif action["type"] == ActionTypes.RollDice:
            roll_value = self.roll_dice()
            self.dice_rolled_this_turn = True
            if roll_value == 7:
                self.can_move_robber = True
        elif action["type"] == ActionTypes.EndTurn:
            self.can_move_robber = False
            self.dice_rolled_this_turn = False
            self.played_development_card_this_turn = False
            self.update_players_go()
            self.turn += 1
            self.development_cards_bought_this_turn = []
        elif action["type"] == ActionTypes.MoveRobber:
            self.board.move_robber(self.board.tiles[action["tile"]])
            self.can_move_robber = False
            self.just_moved_robber = True
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
            self.just_moved_robber = False
        elif action["type"] == ActionTypes.PlayDevelopmentCard:
            player.hidden_cards.remove(action["card"])
            player.visible_cards.append(action["card"])
            self.played_development_card_this_turn = True
            if action["card"] == DevelopmentCard.VictoryPoint:
                player.victory_points += 1
            elif action["card"] == DevelopmentCard.Knight:
                self.can_move_robber = True
                self.update_largest_army()
            elif action["card"] == DevelopmentCard.RoadBuilding:
                self.road_building_active[0] = True
                self.road_building_active[1] = 0
                self.must_use_development_card_ability = True
            elif action["card"] == DevelopmentCard.Monopoly:
                resource = action["resource"]
                for other_player in self.players.values():
                    if other_player.id == player.id:
                        continue
                    res_count = other_player.resources[resource]
                    other_player.resources[resource] = 0
                    player.resources[resource] += res_count
            elif action["card"] == DevelopmentCard.YearOfPlenty:
                for resource in [action["resource_1"], action["resource_2"]]:
                    self.resource_bank[resource] -= 1
                    player.resources[resource] += 1
        elif action["type"] == ActionTypes.BuyDevelopmentCard:
            player.resources[Resource.Sheep] -= 1
            player.resources[Resource.Ore] -= 1
            player.resources[Resource.Wheat] -= 1
            self.resource_bank[Resource.Sheep] += 1
            self.resource_bank[Resource.Ore] += 1
            self.resource_bank[Resource.Wheat] += 1
            player.hidden_cards.append(self.development_cards_pile.pop())
            self.development_cards_bought_this_turn.append(copy.copy(player.hidden_cards[-1]))
        elif action["type"] == ActionTypes.ExchangeResource:
            desired_resource = action["desired_resource"]
            trade_resource = action["trading_resource"]
            player.resources[desired_resource] += 1
            player.resources[trade_resource] -= action["exchange_rate"]
            self.resource_bank[trade_resource] += action["exchange_rate"]
            self.resource_bank[desired_resource] -= 1
        elif action["type"] == ActionTypes.ProposeTrade:
            self.must_respond_to_trade = True
            self.proposed_trade = action.copy()
        elif action["type"] == ActionTypes.RespondToOffer:
            if action["response"] == "accept":
                trade = self.proposed_trade
                p1 = trade["player_proposing"]
                p2 = trade["target_player"]
                for res in trade["player_proposing_res"]:
                    self.players[p1].resources[res] -= 1
                    self.players[p2].resources[res] += 1
                for res in trade["target_player_res"]:
                    self.players[p1].resources[res] += 1
                    self.players[p2].resources[res] -= 1
            self.must_respond_to_trade = False
            self.proposed_trade = None

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

    def update_longest_road(self, player_id):
        player_edges = []
        for edge in self.board.edges:
            if edge.road is not None and edge.road == player_id:
                player_edges.append([edge.corner_1.id, edge.corner_2.id])

        G = defaultdict(list)
        for (s, t) in player_edges:
            G[s].append(t)
            G[t].append(s)

        all_paths = list(chain.from_iterable(DFS(G, n) for n in set(G)))
        max_path_len = max(len(p) - 1 for p in all_paths)

        longest_road_update = {
            "player": player_id,
            "count": max_path_len
        }
        if self.longest_road is None and max_path_len >= 5:
            self.longest_road = longest_road_update
            self.players[player_id].victory_points += 2
        else:
            if self.longest_road is not None:
                if self.longest_road["player"] == player_id:
                    self.longest_road = longest_road_update
                else:
                    if max_path_len > self.longest_road["count"]:
                        self.players[self.longest_road["player"]].victory_points -= 2
                        self.longest_road = longest_road_update
                        self.players[player_id].victory_points += 2