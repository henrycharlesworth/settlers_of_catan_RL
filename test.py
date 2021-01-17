from game.components.board import Board
from ui.display import Display

from game.enums import PlayerId

import numpy as np
import random
np.random.seed(12)
random.seed(12)

class Game(object):
    def __init__(self):
        self.board = Board()

game = Game()

game.board.insert_settlement(PlayerId.Blue, game.board.corners[0], initial_placement=True)
game.board.insert_city(PlayerId.Blue, game.board.corners[0])

game.board.insert_settlement(PlayerId.Orange, game.board.corners[20], initial_placement=True)
game.board.insert_city(PlayerId.Orange, game.board.corners[20])

game.board.insert_road(PlayerId.Orange, game.board.corners[20].corner_neighbours[0][1])

display = Display(game)
display.render()



print("OK?")