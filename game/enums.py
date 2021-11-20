from enum import IntEnum
import copy

class BuildingType(IntEnum):
    Settlement = 0
    City = 1

class PlayerId(IntEnum):
    White = 1
    Blue = 2
    Orange = 3
    Red = 4

class Terrain(IntEnum):
    Desert = 0
    Hills = 1
    Forest = 2
    Mountains = 3
    Pastures = 4
    Fields = 5

class Resource(IntEnum):
    Empty = 0
    Brick = 1
    Wood = 2
    Ore = 3
    Sheep = 4
    Wheat = 5

class DevelopmentCard(IntEnum):
    Knight = 0
    VictoryPoint = 1
    YearOfPlenty = 2
    RoadBuilding = 3
    Monopoly = 4

class ActionTypes(IntEnum):
    PlaceSettlement = 0
    PlaceRoad = 1
    UpgradeToCity = 2
    BuyDevelopmentCard = 3
    PlayDevelopmentCard = 4
    ExchangeResource = 5
    ProposeTrade = 6
    RespondToOffer = 7
    MoveRobber = 8
    RollDice = 9
    EndTurn = 10
    StealResource = 11
    DiscardResource = 12

TILE_ADJACENCY_INDS = [
    [[1, "R"], [3, "BL"], [4, "BR"]],
    [[0, "L"], [2, "R"], [4, "BL"], [5, "BR"]],
    [[1, "L"], [5, "BL"], [6, "BR"]],
    [[0, "TR"], [4, "R"], [7, "BL"], [8, "BR"]],
    [[0, "TL"], [1, "TR"], [3, "L"], [5, "R"], [8, "BL"], [9, "BR"]],
    [[1, "TL"], [2, "TR"], [4, "L"], [6, "R"], [9, "BL"], [10, "BR"]],
    [[2, "TL"], [5, "L"], [10, "BL"], [11, "BR"]],
    [[3, "TR"], [8, "R"], [12, "BR"]],
    [[3, "TL"], [4, "TR"], [7, "L"], [9, "R"], [12, "BL"], [13, "BR"]],
    [[4, "TL"], [5, "TR"], [8, "L"], [10, "R"], [13, "BL"], [14, "BR"]],
    [[5, "TL"], [6, "TR"], [9, "L"], [11, "R"], [14, "BL"], [15, "BR"]],
    [[6, "TL"], [10, "L"], [15, "BL"]],
    [[7, "TL"], [8, "TR"], [13, "R"], [16, "BR"]],
    [[8, "TL"], [9, "TR"], [12, "L"], [14, "R"], [16, "BL"], [17, "BR"]],
    [[9, "TL"], [10, "TR"], [13, "L"], [15, "R"], [17, "BL"], [18, "BR"]],
    [[10, "TL"], [11, "TR"], [14, "L"], [18, "BL"]],
    [[12, "TL"], [13, "TR"], [17, "R"]],
    [[13, "TL"], [14, "TR"], [16, "L"], [18, "R"]],
    [[14, "TL"], [15, "TR"], [17, "L"]]
]

TILE_NEIGHBOURS = []
for inds in TILE_ADJACENCY_INDS:
    tile_dict = {}
    for ind_lab in inds:
        tile_dict[ind_lab[1]] = ind_lab[0]
    TILE_NEIGHBOURS.append(copy.copy(tile_dict))


"""
making sure we don't recreate extra corners/edges when generating the map:
prev_corner_lookup: corner: [[placed_neighbouring_tile, which corner in that tile]]

prev_edge_lookup: edge: [placed_neighbouring_tile, which edge]
"""
PREV_CORNER_LOOKUP = {
    "T": [["TR", "BL"], ["TL", "BR"]],
    "TR": [["TR", "B"]],
    "TL": [["TL", "B"], ["L", "TR"]],
    "BL": [["L", "BR"]],
    "BR": [[None, None]],
    "B": [[None, None]]
}
PREV_EDGE_LOOKUP = {
    "L": ["L", "R"],
    "TL": ["TL", "BR"],
    "TR": ["TR", "BL"],
    "R": [],
    "BR": [],
    "BL": []
}
CORNER_NEIGHBOURS_IN_TILE = {
    "T": {"TR": "TR", "TL": "TL"},
    "TL": {"BL": "L", "T": "TL"},
    "BL": {"TL": "L", "B": "BL"},
    "B": {"BL": "BL", "BR": "BR"},
    "BR": {"B": "BR", "TR": "R"},
    "TR": {"BR": "R", "T": "TR"}
}

"""Harbour corners/edges - first ind is harbour ind.
Then tile_ind, corner_1 loc, corner_2 loc, edge loc
"""
HARBOUR_CORNER_AND_EDGES = {
    0: [0, "TL", "T", "TL"],
    1: [1, "T", "TR", "TR"],
    2: [6, "T", "TR", "TR"],
    3: [11, "TR", "BR", "R"],
    4: [15, "BR", "B", "BR"],
    5: [17, "BR", "B", "BR"],
    6: [16, "B", "BL", "BL"],
    7: [12, "TL", "BL", "L"],
    8: [3, "TL", "BL", "L"]
}