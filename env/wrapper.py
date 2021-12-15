import numpy as np
import copy

from game.game import Game
from game.enums import ActionTypes, DevelopmentCard, Resource, BuildingType, PlayerId

N_CORNERS = 54
N_EDGES = 72
N_TILES = 19

class EnvWrapper(object):
    def __init__(self, interactive=False, max_actions_per_turn=None, max_proposed_trades_per_turn = 4,
                 validate_actions=True, debug_mode=False, win_reward=500, dense_reward=False, policies=None):
        if max_actions_per_turn is None:
            self.max_actions_per_turn = np.inf
        else:
            self.max_actions_per_turn = max_actions_per_turn
        self.max_proposed_trades_per_turn = max_proposed_trades_per_turn
        """
        can turn validate actions off to increase speed slightly. But if you send invalid
        actions it will probably fuck everything up.
        """
        self.validate_actions = validate_actions
        self.game = Game(interactive=interactive, debug_mode=debug_mode, policies=policies)

        self.win_reward = win_reward
        self.dense_reward = dense_reward
        self.reward_annealing_factor = 1.0

    def reset(self):
        self.game.reset()
        self.winner = None
        self.curr_vps = {PlayerId.White: 0, PlayerId.Red: 0, PlayerId.Blue: 0, PlayerId.Orange: 0}
        return self._get_obs()

    def step(self, action):
        translated_action = self._translate_action(action)
        if self.validate_actions:
            valid_action, error = self.game.validate_action(translated_action)
            if valid_action == False:
                raise RuntimeError(error)
        message = self.game.apply_action(translated_action)

        obs = self._get_obs()

        done, reward = self._get_done_and_rewards(action)

        info = {"log": message}

        return obs, reward, done, info

    def _get_obs(self):
        if self.game.players_need_to_discard:
            player = self.game.players[self.game.players_to_discard[0]]
        elif self.game.must_respond_to_trade:
            player = self.game.players[self.game.proposed_trade["target_player"]]
        else:
            player = self.game.players[self.game.players_go]

        obs = {
            "proposed_trade": np.zeros((12,)),
            "current_resources": np.zeros((6,)),
            "player_id": player.id
        }
        if self.game.proposed_trade is not None:
            for res in self.game.proposed_trade["player_proposing_res"]:
                obs["proposed_trade"][res] = 1.0
            for res in self.game.proposed_trade["target_player_res"]:
                obs["proposed_trade"][res + 5] = 1.0
        for res in [Resource.Brick, Resource.Wood, Resource.Ore, Resource.Sheep, Resource.Wheat]:
            obs["current_resources"][res] = self.game.players[player.id].resources[res]

        obs["tile_representations"] = self._get_tile_features(player)

        obs["current_player_main"], obs["current_player_played_dev"], obs["current_player_hidden_dev"] = \
            self._get_player_inputs(player, "current")

        for target_p in ["next", "next_next", "next_next_next"]:
            obs[target_p+"_player_main"], obs[target_p + "_player_played_dev"], _ = self._get_player_inputs(
                player, target_p
            )

        return obs

    def _get_done_and_rewards(self, action):
        done = False
        rewards = {player: 0 for player in [PlayerId.White, PlayerId.Red, PlayerId.Blue, PlayerId.Orange]}
        for id, player in self.game.players.items():
            if player.victory_points >= 10:
                done = True
                self.winner = player
        updated_vps = {}
        for player_id in [PlayerId.Blue, PlayerId.Orange, PlayerId.Red, PlayerId.White]:
            updated_vps[player_id] = self.game.players[player_id].victory_points
            if self.dense_reward:
                rewards[player_id] += 5 * (updated_vps[player_id] - self.curr_vps[player_id])
                if action[0] == ActionTypes.PlayDevelopmentCard:
                    rewards[player_id] += 5
                if action[0] == ActionTypes.MoveRobber:
                    rewards[player_id] += 1
                if action[0] == ActionTypes.DiscardResource:
                    rewards[player_id] -= 0.3
                if action[0] == ActionTypes.UpgradeToCity:
                    rewards[player_id] += 2.5

                rewards[player_id] *= self.reward_annealing_factor
        self.curr_vps = updated_vps

        if done:
            rewards[self.winner.id] += self.win_reward

        return done, rewards

    def _translate_action(self, action):
        players_go = self.game.players_go
        translated_action = {}
        action_type = action[0]
        translated_action["type"] = action_type

        if action_type == ActionTypes.PlaceSettlement or action_type == ActionTypes.UpgradeToCity:
            translated_action["corner"] = action[1]
        elif action_type == ActionTypes.PlaceRoad:
            if action[2] == N_EDGES:
                translated_action["edge"] = None #dummy edge
            else:
                translated_action["edge"] = action[2]
        elif action_type == ActionTypes.MoveRobber:
            translated_action["tile"] = action[3]
        elif action_type == ActionTypes.StealResource:
            target_player = action[6] #in form next, next-next.
            if target_player == 0:
                target_id = "next"
            elif target_player == 1:
                target_id = "next_next"
            elif target_player == 2:
                target_id = "next_next_next"
            else:
                raise ValueError
            translated_action["target"] = self.game.players[players_go].inverse_player_lookup[target_id]
        elif action_type == ActionTypes.PlayDevelopmentCard:
            card_type = int(action[4])
            translated_action["card"] = card_type
            if card_type == DevelopmentCard.Monopoly:
                translated_action["resource"] = self._head_out_to_res(action[9]) #need to properly map head out to resource intenum
            elif card_type == DevelopmentCard.YearOfPlenty:
                translated_action["resource_1"] = self._head_out_to_res(action[9])
                translated_action["resource_2"] = self._head_out_to_res(action[10])
        elif action_type == ActionTypes.ExchangeResource:
            exchange_res = self._head_out_to_res(action[9])
            desired_res = self._head_out_to_res(action[10])
            translated_action["exchange_rate"] = self._get_best_exchange_rate(exchange_res, self.game.players[players_go])
            translated_action["desired_resource"] = desired_res
            translated_action["trading_resource"] = exchange_res
        elif action_type == ActionTypes.ProposeTrade:
            translated_action = self._parse_trade(action, translated_action, self.game.players[players_go])
        elif action_type == ActionTypes.RespondToOffer:
            if action[5] == 0:
                translated_action["response"] = "accept"
            elif action[5] == 1:
                translated_action["response"] = "reject"
            else:
                raise ValueError
        elif action_type == ActionTypes.DiscardResource:
            translated_action["resources"] = [self._head_out_to_res(action[11])]

        return translated_action

    def get_action_masks(self):
        player = self.game.players[self.game.players_go]
        num_actions = len(ActionTypes)

        valid_actions = [
            np.zeros((num_actions,)),
            np.ones((3, N_CORNERS,)), #place settlement/city head
            np.ones((N_EDGES+1,)), #build road head
            np.ones((N_TILES,)), #move robber head
            np.ones((len(DevelopmentCard),)), #play dev card head
            np.ones((2,)), #accept/reject head
            np.ones((3, 3)), #player head
            np.ones((6,)), #propose trade head
            np.ones((6,)), #propose trade receive head
            np.ones((4, 5)), #exchange this res head
            np.ones((5,)), #receive this res head
            np.ones((5,)) #discard resource head
        ]
        if self.game.players_need_to_discard:
            player = self.game.players[self.game.players_to_discard[0]]
            valid_actions[0][ActionTypes.DiscardResource] = 1.0
            for i, res in enumerate([Resource.Brick, Resource.Wood, Resource.Ore, Resource.Sheep, Resource.Wheat]):
                if player.resources[res] <= 0:
                    valid_actions[11][i] = 0.0
            return valid_actions

        """Action types"""
        if self.game.initial_placement_phase:
            if self.game.initial_settlements_placed[player.id] == 0 \
                or self.game.initial_settlements_placed[player.id] == 1 and self.game.initial_roads_placed[player.id] == 1:
                valid_actions[0][ActionTypes.PlaceSettlement] = 1.0
                valid_actions[1][0] = self._get_valid_settlement_locations(player, initial_phase=True)
                return valid_actions
            else:
                valid_actions[0][ActionTypes.PlaceRoad] = 1.0
                valid_actions[2] = self._get_valid_road_locations(player)
                return valid_actions

        if self.game.road_building_active[0]:
            valid_actions[0][ActionTypes.PlaceRoad] = 1.0
            valid_actions[2] = self._get_valid_road_locations(player, road_building=True)
            return valid_actions
        elif self.game.just_moved_robber:
            valid_actions[0][ActionTypes.StealResource] = 1.0
            valid_actions[6][1, :] = self._get_valid_steal_targets(player)
            return valid_actions
        elif self.game.must_respond_to_trade:
            valid_actions[0][ActionTypes.RespondToOffer] = 1.0
            target_player = self.game.players[self.game.proposed_trade["target_player"]]
            valid_actions[5] = self._get_valid_accept_reject_offer(target_player)
            return valid_actions
        elif self.game.dice_rolled_this_turn == False:
            valid_actions[0][ActionTypes.RollDice] = 1.0
            if len(player.hidden_cards) > 0 and self.game.played_development_card_this_turn == False:
                valid_dev_card, valid_exch_res = self._get_valid_actions_play_dev_card(player)
                if sum(valid_dev_card) > 0:
                    valid_actions[0][ActionTypes.PlayDevelopmentCard] = 1.0
                    valid_actions[4] = valid_dev_card
                    if valid_exch_res is not None:
                        valid_actions[9][2] = valid_exch_res
                        valid_actions[10] = valid_exch_res
            return valid_actions

        #otherwise we've rolled the dice and have potentially many options available.
        valid_actions[0][ActionTypes.EndTurn] = 1.0
        if self.game.actions_this_turn > self.max_actions_per_turn:
            return valid_actions

        resources = player.resources
        #place settlement
        if resources[Resource.Wheat] > 0 and resources[Resource.Sheep] > 0 and resources[Resource.Wood] > 0 and \
            resources[Resource.Brick] > 0:
            valid_corners = self._get_valid_settlement_locations(player)
            if sum(valid_corners) > 0 and self.game.building_bank["settlements"][player.id] > 0:
                valid_actions[0][ActionTypes.PlaceSettlement] = 1.0
                valid_actions[1][0] = valid_corners
        #upgrade to city
        if resources[Resource.Wheat] >= 2 and resources[Resource.Ore] >= 3 and \
                self.game.building_bank["cities"][player.id] > 0:
            valid_corners = self._get_valid_city_locations(player)
            if sum(valid_corners) > 0:
                valid_actions[0][ActionTypes.UpgradeToCity] = 1.0
                valid_actions[1][1] = valid_corners
        #place road
        if resources[Resource.Wood] > 0 and resources[Resource.Brick] > 0:
            valid_edges = self._get_valid_road_locations(player)
            if sum(valid_edges) > 0:
                valid_actions[0][ActionTypes.PlaceRoad] = 1.0
                valid_actions[2] = valid_edges
        #buy development card
        if resources[Resource.Wheat] > 0 and resources[Resource.Sheep] > 0 and resources[Resource.Ore] > 0:
            if len(self.game.development_cards_pile) > 0:
                valid_actions[0][ActionTypes.BuyDevelopmentCard] = 1.0
        #play development card
        if len(player.hidden_cards) > 0 and self.game.played_development_card_this_turn == False:
            valid_dev_card, valid_exch_res = self._get_valid_actions_play_dev_card(player)
            if sum(valid_dev_card) > 0:
                valid_actions[0][ActionTypes.PlayDevelopmentCard] = 1.0
                valid_actions[4] = valid_dev_card
                if valid_exch_res is not None:
                    valid_actions[9][2] = valid_exch_res
                    valid_actions[10] = valid_exch_res
        #exchange resources
        valid_resources_to_exchange = self._get_valid_exchange_resources(player)
        valid_resources_to_receive = self._get_valid_exchange_receive_resources()
        if sum(valid_resources_to_exchange) > 0 and sum(valid_resources_to_receive) > 0:
            valid_actions[0][ActionTypes.ExchangeResource] = 1.0
            valid_actions[9][0] = valid_resources_to_exchange
            valid_actions[10] = valid_resources_to_receive
        #move robber
        if self.game.can_move_robber:
            valid_tiles = self._get_valid_robber_locations()
            valid_actions[0][ActionTypes.MoveRobber] = 1.0
            valid_actions[3] = valid_tiles
        #propose trade
        total_res = sum(resources.values())
        if self.max_proposed_trades_per_turn is None:
            if total_res > 0:
                valid_actions[0][ActionTypes.ProposeTrade] = 1.0
        else:
            if self.game.trades_proposed_this_turn < self.max_proposed_trades_per_turn and total_res > 0:
                valid_actions[0][ActionTypes.ProposeTrade] = 1.0
        return valid_actions


    def _get_valid_settlement_locations(self, player, initial_phase=False):
        valid_corners = np.zeros((N_CORNERS,))
        for i, corner in enumerate(self.game.board.corners):
            if corner.can_place_settlement(player.id, initial_placement=initial_phase):
                valid_corners[i] = 1.0
        return valid_corners

    def _get_valid_city_locations(self, player):
        valid_corners = np.zeros((N_CORNERS,))
        for i, corner in enumerate(self.game.board.corners):
            if corner.building is not None and corner.building.type == BuildingType.Settlement:
                if corner.building.owner == player.id:
                    valid_corners[i] = 1.0
        return valid_corners

    def _get_valid_robber_locations(self):
        valid_tiles = np.zeros((N_TILES,))
        curr_player = self.game.players_go

        for i, tile in enumerate(self.game.board.tiles):
            valid = False
            for key in tile.corners.keys():
                if tile.corners[key].building is not None and tile.corners[key].building != curr_player:
                    valid = True
                    break
            if valid:
                valid_tiles[i] = 1.0
        return valid_tiles

    def _get_valid_road_locations(self, player, road_building=False):
        valid_edges = np.zeros((N_EDGES+1,))
        placed_edge = False
        after_second_settlement = False
        second_settlement = None
        if self.game.initial_placement_phase:
            if self.game.initial_settlements_placed[self.game.players_go] == 2:
                after_second_settlement = True
                second_settlement = self.game.initial_second_settlement_corners[self.game.players_go]
        for i, edge in enumerate(self.game.board.edges):
            if edge.can_place_road(player.id, after_second_settlement=after_second_settlement,
                                   second_settlement=second_settlement):
                valid_edges[i] = 1.0
                placed_edge = True
        if placed_edge == False:
            if road_building:
                valid_edges[-1] = 1.0  # dummy edge that does nothing
        return valid_edges

    def _get_valid_steal_targets(self, player):
        map = {"next": 0, "next_next": 1, "next_next_next": 2}
        valid_steal_targets = np.zeros((3,))
        robber_tile = self.game.board.robber_tile
        for corner in robber_tile.corners.values():
            if corner is not None and corner.building is not None:
                if corner.building.owner == player.id:
                    continue
                label = player.player_lookup[corner.building.owner]
                valid_steal_targets[map[label]] = 1.0
        return valid_steal_targets

    def _get_valid_accept_reject_offer(self, player):
        valid_action = np.ones((2,))
        trade = self.game.proposed_trade
        check_resources = player.resources.copy()
        have_res = True
        for res in trade["target_player_res"]:
            check_resources[res] -= 1
            if check_resources[res] < 0:
                have_res = False
                break
        if have_res == False:
            valid_action[0] = 0.0
        return valid_action


    def _get_valid_actions_play_dev_card(self, player):
        valid_development_cards = np.zeros((len(DevelopmentCard),))
        for card in [DevelopmentCard.VictoryPoint, DevelopmentCard.Monopoly, DevelopmentCard.YearOfPlenty, \
                     DevelopmentCard.RoadBuilding, DevelopmentCard.Knight]:
            card_count = player.hidden_cards.count(card)
            if card_count > 0:
                if self.game.development_cards_bought_this_turn.count(card) < card_count:
                    if card == DevelopmentCard.YearOfPlenty:
                        bank_resources = sum(self.game.resource_bank.values())
                        if bank_resources > 0:
                            valid_development_cards[card] = 1.0
                    else:
                        valid_development_cards[card] = 1.0
        if valid_development_cards[DevelopmentCard.YearOfPlenty] > 0:
            valid_exchange_res = np.zeros((5,))
            for i, res in enumerate([Resource.Brick, Resource.Wood, Resource.Ore, Resource.Sheep, Resource.Wheat]):
                if self.game.resource_bank[res] > 0:
                    valid_exchange_res[i] = 1.0
        else:
            valid_exchange_res = None
        return valid_development_cards, valid_exchange_res

    def _get_valid_exchange_resources(self, player):
        valid_resources = np.zeros((len(Resource)-1,))
        res_map = {Resource.Brick: 0, Resource.Wood: 1, Resource.Ore: 2, Resource.Sheep: 3, Resource.Wheat: 4}
        resources = player.resources
        for harbour in player.harbours.values():
            if harbour.resource is None:
                for i, res in enumerate([Resource.Brick, Resource.Wood, Resource.Ore, Resource.Sheep, Resource.Wheat]):
                    if resources[res] >= 3:
                        valid_resources[i] = 1.0
            else:
                if resources[harbour.resource] >= 2:
                    valid_resources[res_map[harbour.resource]] = 1.0
        for i, res in enumerate([Resource.Brick, Resource.Wood, Resource.Ore, Resource.Sheep, Resource.Wheat]):
            if resources[res] >= 4:
                valid_resources[i] = 1.0 #can exchange at 4:1 with no harbour.
        return valid_resources

    def _get_valid_exchange_receive_resources(self):
        valid_resources = np.zeros((len(Resource)-1,))
        for i, res in enumerate([Resource.Brick, Resource.Wood, Resource.Ore, Resource.Sheep, Resource.Wheat]):
            if self.game.resource_bank[res] > 0:
                valid_resources[i] = 1.0
        return valid_resources

    def _head_out_to_res(self, action):
        if action == 0:
            return Resource.Brick
        elif action == 1:
            return Resource.Wood
        elif action == 2:
            return Resource.Ore
        elif action == 3:
            return Resource.Sheep
        elif action == 4:
            return Resource.Wheat
        else:
            raise ValueError

    def _get_best_exchange_rate(self, res, player):
        best_exchange_rate = 4
        for harbour in player.harbours.values():
            if harbour.resource is None:
                if best_exchange_rate >= 3:
                    best_exchange_rate = 3
            else:
                if harbour.resource == res:
                    best_exchange_rate = 2
                    break
        return best_exchange_rate

    def _parse_trade(self, head_outputs, translated_action, player):
        translated_action["player_proposing"] = player.id
        player_head = head_outputs[6]
        map = {0: "next", 1: "next_next", 2: "next_next_next"}
        other_player_label = map[int(player_head)]
        other_player_id = player.inverse_player_lookup[other_player_label]
        translated_action["target_player"] = other_player_id
        player_res_out = head_outputs[7]
        target_player_res_out = head_outputs[8]
        player_proposed_res = []
        target_proposed_res = []
        for i in player_res_out:
            if i == 0:
                break
            elif i == 1:
                res = Resource.Brick
            elif i == 2:
                res = Resource.Wood
            elif i == 3:
                res = Resource.Ore
            elif i == 4:
                res = Resource.Sheep
            elif i == 5:
                res = Resource.Wheat
            else:
                raise ValueError
            player_proposed_res.append(res)

        for i in target_player_res_out:
            if i == 0:
                break
            elif i == 1:
                res = Resource.Brick
            elif i == 2:
                res = Resource.Wood
            elif i == 3:
                res = Resource.Ore
            elif i == 4:
                res = Resource.Sheep
            elif i == 5:
                res = Resource.Wheat
            else:
                raise ValueError
            target_proposed_res.append(res)
        translated_action["player_proposing_res"] = player_proposed_res
        translated_action["target_player_res"] = target_proposed_res
        return translated_action

    def render(self):
        return self.game.render()

    def _get_tile_features(self, player):
        tile_features = []
        for tile in self.game.board.tiles:
            contains_robber = np.array([tile.contains_robber], dtype=np.float32)
            value = np.zeros((11,))
            value[tile.value - 2] = 1.0
            resource = np.zeros((6,))
            resource[tile.resource] = 1.0
            corners = []
            for corner_id in ["T", "TL", "BL", "B", "BR", "TR"]:
                corner = tile.corners[corner_id]
                building = np.zeros((3,))
                if corner.building is None:
                    building[0] = 1.0
                elif corner.building.type == BuildingType.Settlement:
                    building[1] = 1.0
                elif corner.building.type == BuildingType.City:
                    building[2] = 1.0
                owner = np.zeros((4,))
                if corner.building is not None:
                    building_owner = corner.building.owner
                    if building_owner == player.id:
                        owner[0] = 1.0
                    elif building_owner == player.inverse_player_lookup["next"]:
                        owner[1] = 1.0
                    elif building_owner == player.inverse_player_lookup["next_next"]:
                        owner[2] = 1.0
                    elif building_owner == player.inverse_player_lookup["next_next_next"]:
                        owner[3] = 1.0
                corners.append(building)
                corners.append(owner)
            tile_feature = np.concatenate((contains_robber, value, resource, *corners))
            tile_features.append(tile_feature)
        return tile_features

    def _get_player_inputs(self, player, target_player_id="current"):
        if target_player_id == "current":
            target_player = player
        else:
            target_player = self.game.players[player.inverse_player_lookup[target_player_id]]

        if target_player_id != "current":
            other_player_id = np.zeros((3,))
            if target_player_id == "next":
                other_player_id[0] = 1.0
            elif target_player_id == "next_next":
                other_player_id[1] = 1.0
            elif target_player_id == "next_next_next":
                other_player_id[2] = 1.0
            else:
                raise ValueError

        """resources"""
        if target_player_id != "current":
            min_resources = []
            max_resources = []
        else:
            resources = []

        for res in [Resource.Wood, Resource.Brick, Resource.Wheat, Resource.Ore, Resource.Sheep]:
            if target_player_id == "current":
                res_count = np.zeros((8,))
                res_num = target_player.resources[res]
                if res_num < 5:
                    res_count[res_num] = 1.0
                elif res_num < 8:
                    res_count[5] = 1.0
                elif res_num < 11:
                    res_count[6] = 1.0
                else:
                    res_count[7] = 1.0
                resources.append(res_count)
            else:
                min_res_count = np.zeros((8,))
                max_res_count = np.zeros((8,))
                min_res_num = player.opponent_min_res[target_player_id][res]
                max_res_num = player.opponent_max_res[target_player_id][res]
                if min_res_num < 5:
                    min_res_count[min_res_num] = 1.0
                elif min_res_num < 8:
                    min_res_count[5] = 1.0
                elif min_res_num < 11:
                    min_res_count[6] = 1.0
                else:
                    min_res_count[7] = 1.0
                min_resources.append(min_res_count)
                if max_res_num < 5:
                    max_res_count[max_res_num] = 1.0
                elif max_res_num < 8:
                    max_res_count[5] = 1.0
                elif max_res_num < 11:
                    max_res_count[6] = 1.0
                else:
                    max_res_count[7] = 1.0
                max_resources.append(max_res_count)

        """victory points"""
        victory_points = np.zeros((10,))
        vps = target_player.victory_points
        if vps < 10:
            victory_points[vps] = 1.0
        else:
            victory_points[-1] = 1.0

        """resource access"""
        res_access = {}
        for res in [Resource.Wood, Resource.Brick, Resource.Wheat, Resource.Ore, Resource.Sheep]:
            res_access[res] = np.zeros((10,))
        for corner, building_type in target_player.buildings.items():
            for tile in self.game.board.corners[corner].adjacent_tiles:
                if tile is not None and tile.value != 7:
                    if building_type == BuildingType.Settlement:
                        to_add = 1
                    elif building_type == BuildingType.City:
                        to_add = 2
                    if tile.value <= 6:
                        ind = tile.value - 2
                    else:
                        ind = tile.value - 3
                    res_access[tile.resource][ind] += to_add

        """longest road"""
        longest_road = np.zeros((2,))
        if self.game.longest_road is not None:
            if self.game.longest_road["player"] == target_player.id:
                longest_road[0] = 1.0
                longest_road[1] = self.game.longest_road["count"] / 8.0
            else:
                if target_player.id in self.game.current_longest_path:
                    longest_road[1] = self.game.current_longest_path[target_player.id] / 8.0

        """largest army"""
        largest_army = np.zeros((2,))
        if self.game.largest_army is not None:
            if self.game.largest_army["player"] == target_player.id:
                largest_army[0] = 1.0
        largest_army[1] = self.game.current_army_size[target_player.id] / 4.0

        """
        harbours
        """
        harbour_access = np.zeros((6,))
        for harbour in target_player.harbours.values():
            if harbour.exchange_value == 3:
                harbour_access[0] = 1.0
            else:
                harbour_access[harbour.resource] = 1.0

        """
        Development cards
        """
        if len(target_player.visible_cards) == 0:
            played_cards = np.zeros((1,), dtype=int)
        else:
            played_cards = np.zeros((len(target_player.visible_cards),), dtype=int)
        for i, card in enumerate(target_player.visible_cards):
            played_cards[i] = card + 1

        if target_player_id == "current":
            if len(target_player.hidden_cards) == 0:
                hidden_cards = np.zeros((1,), dtype=int)
            else:
                hidden_cards = np.zeros((len(target_player.hidden_cards),), dtype=int)
            for i, card in enumerate(target_player.hidden_cards):
                hidden_cards[i] = card + 1

            """bank resources"""
            bank_resources = []
            for res in [Resource.Wood, Resource.Brick, Resource.Wheat, Resource.Ore, Resource.Sheep]:
                bank_res = np.zeros((7,))
                num_res = self.game.resource_bank[res]
                if num_res <= 2:
                    bank_res[num_res] = 1.0
                elif num_res <=5:
                    bank_res[3] = 1.0
                elif num_res <= 7:
                    bank_res[4] = 1.0
                elif num_res <= 10:
                    bank_res[5] = 1.0
                else:
                    bank_res[6] = 1.0
                bank_resources.append(bank_res)

            """bank dev cards"""
            dev_cards_left = len(self.game.development_cards_pile)
            dev_cards_bank = np.zeros((7,))
            if dev_cards_left <= 2:
                dev_cards_bank[dev_cards_left] = 1.0
            elif dev_cards_left <=5:
                dev_cards_bank[3] = 1.0
            elif dev_cards_left <= 7:
                dev_cards_bank[4] = 1.0
            elif dev_cards_left <= 10:
                dev_cards_bank[5] = 1.0
            else:
                dev_cards_bank[6] = 1.0

        else:
            hidden_cards = None
            num_hidden_dev_cards = np.zeros((6,))
            num_cards = len(target_player.hidden_cards)
            if num_cards <= 4:
                num_hidden_dev_cards[num_cards] = 1.0
            else:
                num_hidden_dev_cards[-1] = 1.0


        if target_player_id == "current":
            main_inputs = np.concatenate(
                (*resources, victory_points, *list(res_access.values()), longest_road, largest_army, harbour_access,
                 *bank_resources, dev_cards_bank)
            )
        else:
            main_inputs = np.concatenate(
                (*min_resources, *max_resources, victory_points, *list(res_access.values()), longest_road,
                 largest_army, harbour_access, other_player_id, num_hidden_dev_cards)
            )

        return main_inputs, played_cards, hidden_cards

    def save_state(self):
        return copy.deepcopy({
            "state": self.game.save_current_state(),
            "vps": copy.deepcopy(self.curr_vps),
            "winner": self.winner
        })

    def restore_state(self, state):
        self.game.restore_state(state["state"])
        self.curr_vps = state["vps"]
        self.winner = state["winner"]
