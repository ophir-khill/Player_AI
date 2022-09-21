"""
MiniMax Player with AlphaBeta pruning with heavy heuristic
"""
from players.AbstractPlayer import AbstractPlayer
from players.MinimaxPlayer import Player as MiniMax

# TODO: you can import more modules, if needed


class Player(AbstractPlayer):
    def __init__(self, game_time):
        AbstractPlayer.__init__(self, game_time)  # keep the inheritance of the parent's (AbstractPlayer) __init__()
        self.player1 = MiniMax(game_time)
        self.player1.enable_heavy_player()

    def set_game_params(self, board):
        """Set the game parameters needed for this player.
        This function is called before the game starts.
        (See GameWrapper.py for more info where it is called)
        input:
            - board: np.array, of the board.
        No output is expected.
        """
        self.player1.set_game_params(board)

    def make_move(self, time_limit):
        """Make move with this Player.
        input:
            - time_limit: float, time limit for a single turn.
        output:
            - direction: tuple, specifing the Player's movement
        """
        # TODO: erase the following line and implement this function.
        return self.player1.make_move(time_limit)

    def set_rival_move(self, move):
        """Update your info, given the new position of the rival.
        input:
            - move: tuple, the new position of the rival.
        No output is expected
        """
        # TODO: erase the following line and implement this function.
        self.player1.set_rival_move(move)

    ########## helper functions in class ##########
    # TODO: add here helper functions in class, if needed

    ########## helper functions for AlphaBeta algorithm ##########
    # TODO: add here the utility, succ, and perform_move functions used in AlphaBeta algorithm