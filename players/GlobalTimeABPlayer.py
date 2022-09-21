"""
MiniMax Player with AlphaBeta pruning and global time
"""
from players.AbstractPlayer import AbstractPlayer
#TODO: you can import more modules, if needed
import numpy as np
import utils
import time
import SearchAlgos
import copy
from players.AlphabetaPlayer import Player as AlphaBeta

class Player(AbstractPlayer):
    def __init__(self, game_time):
        AbstractPlayer.__init__(self, game_time) # keep the inheritance of the parent's (AbstractPlayer) __init__()
        self.global_player = AlphaBeta(game_time)

    def set_game_params(self, board):
        """Set the game parameters needed for this player.
        This function is called before the game starts.
        (See GameWrapper.py for more info where it is called)
        input:
            - board: np.array, of the board.
        No output is expected.
        """
        self.global_player.set_game_params(board)


    def make_move(self, time_limit):
        """Make move with this Player.
        input:
            - time_limit: float, time limit for a single turn.
        output:
            - direction: tuple, specifing the Player's movement
        """
        if self.global_player.player1.turn_count == 1:
            turn_time_limit = 0.09
        elif self.global_player.player1.turn_count < 10:
            turn_time_limit = self.game_time * 0.03   # 5 turns (player turns) - 85% of game_time is left
        elif self.global_player.player1.turn_count < 18:
            turn_time_limit = self.game_time * 0.1  # 9 turns (player turns) - 56% of game_time is left
        elif self.game_time > 10:
            turn_time_limit = self.game_time * 0.1  # 15 turns (player turns) - 10% of game_time is left
        elif self.game_time > 2:
            turn_time_limit = 1
        else:
            # simple play
            turn_time_limit = 0.09

        start = time.time()
        move = self.global_player.make_move(turn_time_limit)
        self.game_time -= time.time() - start
        return move



    def set_rival_move(self, move):
        """Update your info, given the new position of the rival.
        input:
            - move: tuple, the new position of the rival.
        No output is expected
        """
        #TODO: erase the following line and implement this function.

        self.global_player.set_rival_move(move)



    ########## helper functions in class ##########
    #TODO: add here helper functions in class, if needed


    ########## helper functions for AlphaBeta algorithm ##########
    #TODO: add here the utility, succ, and perform_move functions used in AlphaBeta algorithm