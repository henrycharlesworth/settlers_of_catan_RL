import numpy as np

from game.game import Game
from game.enums import ActionTypes, DevelopmentCard, Resource, BuildingType

N_CORNERS = 54
N_EDGES = 72
N_TILES = 19

class EnvWrapper(object):
    def __init__(self, interactive=False, max_actions_per_turn=None, max_proposed_trades_per_turn = None,
                 validate_actions=True, debug_mode=False):
        """
        ultimately should input policies and specify which are actually being recorded - only return obs etc for those.
        BUT - this will really slow down getting to the next observation (if stepping in parallel will have to wait
        for all envs - some of which will be stepping through other players gos). Almost makes you want to go
        asynchronous... ugh. Or we just always play most recent policy? IDK :(

        could maybe collect batches within each process and return full batch, which then has to be mashed together...
        I think this is best approach though - we have to periodically update weights anyway, so this will be efficient-ish.
        Can also do batch processing of returns/advantages so that batches can just be appended together on the GPU end.
        YAHYAHYAH. obviously makes code more difficult but meh.

        Maybe make this a wrapper on top of this "normal" environment. Probably a good idea.
        """
        if max_actions_per_turn is None:
            self.max_actions_per_turn = np.inf
        else:
            self.max_actions_per_turn = max_actions_per_turn
        self.max_proposed_trades_per_turn = max_proposed_trades_per_turn
        self.validate_actions = validate_actions
        self.game = Game(interactive=interactive, debug_mode=debug_mode)
        """
        can turn validate actions off to increase speed slightly. But if you send invalid
        actions it will probably fuck everything up.
        """

    def reset(self):
        self.game.reset()
        self.winner = None
        main_obs, custom_inputs = self._get_obs()
        obs = {
            "main": main_obs,
            "custom_inputs": custom_inputs,
            "players_go": self.game.players_go
        }
        return obs

    def step(self, action):
        translated_action = self._translate_action(action)
        if self.validate_actions:
            valid_action, error = self.game.validate_action(translated_action)
            if valid_action == False:
                raise RuntimeError(error)
        self.game.apply_action(translated_action)

        main_obs, custom_inputs = self._get_obs()
        obs = {
            "main": main_obs,
            "custom_inputs": custom_inputs,
            "players_go": self.game.players_go
        }

        done, reward = self._get_done_and_rewards()

        info = {}

        return obs, reward, done, info

    def _get_obs(self):
        """temporary"""
        main_obs = np.random.randn(32)
        custom_inputs = {
            "proposed_trade": np.zeros((12,)),
            "current_resources": np.zeros((6,))
        }
        if self.game.proposed_trade is not None:
            for res in self.game.proposed_trade["player_proposing_res"]:
                custom_inputs["proposed_trade"][res] = 1.0
            for res in self.game.proposed_trade["target_player_res"]:
                custom_inputs["proposed_trade"][res + 5] = 1.0
        for res in [Resource.Brick, Resource.Wood, Resource.Ore, Resource.Sheep, Resource.Wheat]:
            custom_inputs["current_resources"][res] = self.game.players[self.game.players_go].resources[res]
        return main_obs, custom_inputs

    def _get_done_and_rewards(self):
        """temporary"""
        done = False
        for id, player in self.game.players.items():
            if player.victory_points >= 10:
                done = True
                self.winner = player
        """deal with rewards etc later. Will be a little bit more complicated as need to save rewards
        at end of turn and potentially "cycle back" and give them. fine though."""
        return done, 0.0

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
            np.ones((5,)) #receive this res head
        ]
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
        if sum(valid_resources_to_exchange) > 0:
            valid_actions[0][ActionTypes.ExchangeResource] = 1.0
            valid_actions[9][0] = valid_resources_to_exchange
            valid_actions[10] = self._get_valid_exchange_receive_resources()
        #move robber - no tile constraints
        if self.game.can_move_robber:
            valid_actions[0][ActionTypes.MoveRobber] = 1.0
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

    def _get_tile_features(self):
        """
        NEEDS TESTING
        :return:
        """
        if self.game.must_respond_to_trade:
            player = self.game.players[self.game.proposed_trade["target_player"]]
        else:
            player = self.game.players[self.game.players_go]
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
