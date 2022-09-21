"""
MiniMax Player
"""
from players.AbstractPlayer import AbstractPlayer
# TODO: you can import more modules, if needed
import numpy as np
import utils
import time
import SearchAlgos
import copy


class Player(AbstractPlayer):
    def __init__(self, game_time):
        AbstractPlayer.__init__(self, game_time)  # keep the inheritance of the parent's (AbstractPlayer) __init__()
        self.turn_count = 0  # increase when make moves
        self.player_pos = np.full(9, -1)
        self.rival_pos = np.full(9, -1)
        self.player_index = 1
        self.rival_index = 2
        self.AlphaBeta = False
        self.heavy_player = False
        self.light_player = False
        self.is_switched = 0
        # TODO: initialize more fields, if needed, and the AlphaBeta algorithm from SearchAlgos.py

    def set_game_params(self, board):
        """Set the game parameters needed for this player.
        This function is called before the game starts.
        (See GameWrapper.py for more info where it is called)
        input:
            - board: np.array, of the board.
        No output is expected.
        """
        # TODO: erase the following line and implement this function.
        self.board = board

    def make_move(self, time_limit):
        """Make move with this Player.
        input:
            - time_limit: float, time limit for a single turn.
        output:
            - direction: tuple, specifing the Player's movement
            :return: move = (pos, soldier, dead_opponent_pos)
        """
        start = time.time()
        end = start + time_limit
        max_value = -np.inf
        # direction (the new cell, soldier - 10 sized array, rival dead soldier - 10 sized array)
        best_move = (-1, -1, -1)
        depth = 1
        # state = (self.board, self.turn_count, self.player_pos, self.rival_pos, 1, best_move)
        if self.light_player:
            minimax = SearchAlgos.AlphaBeta(calculate_simple_heuristic, succ, None, is_goal)
        elif self.AlphaBeta or self.heavy_player:
            minimax = SearchAlgos.AlphaBeta(calculate_state_heuristic, succ, None, is_goal)
        else:
            minimax = SearchAlgos.MiniMax(calculate_state_heuristic, succ, None, is_goal)
        end_phase = 0
        time_condition = 0
        # while end - time.time() > 63 * end_phase:
        # phase 1 : (24-turn)*(24-turn-(depth-1))*...*() => (24-turn-(depth-1)) * int(turn/2)
        if self.heavy_player:
            value, best_move = minimax.search((copy.deepcopy(self), best_move), 2, True)
            self.update_move(best_move)
            return best_move
        elif self.light_player:
            ## change depth
            value, best_move = minimax.search((copy.deepcopy(self), best_move), 3, True)
            self.update_move(best_move)
            return best_move
        else:
            if time_limit < 0.1:
                if self.turn_count < 18:
                    move = self.simple_stage_1_move()
                    self.update_move(move)
                    return move
                else:
                    move = self.simple_stage_2_move()
                    self.update_move(move)
                    return move
            else:
                while end - time.time() > time_condition * end_phase:
                    start_phase = time.time()
                    value, direction = minimax.search((copy.deepcopy(self), best_move), depth, True)
                    if value > max_value:
                        best_move = direction
                        max_value = value
                    end_phase = time.time() - start_phase
                    phase_1_end = (24 - self.turn_count - depth) + 2 * (np.count_nonzero(self.rival_pos > -1))
                    phase_2_end = 2.5 * (np.count_nonzero(self.player_pos > -1)) + 1 * (np.count_nonzero(
                        self.rival_pos > -1))  ## 2.5 is avg branch of soldier + (avg soldiers able to kill) * kill
                    time_condition = phase_1_end if self.turn_count < 18 else phase_2_end
                    depth += 1
                # update self values
                self.update_move(best_move)
                return best_move

    def set_rival_move(self, move):
        """Update your info, given the new position of the rival.
        input:
            - move: tuple, the new position of the rival.
        No output is expected
        """
        # direction (the new cell, soldier - 10 sized array, rival dead soldier - 10 sized array)
        rival_soldier_cell, rival_soldier, my_dead_pos = move
        if self.turn_count < 18:
            self.board[rival_soldier_cell] = self.rival_index
            self.rival_pos[rival_soldier] = rival_soldier_cell
        else:
            rival_prev_pos = self.rival_pos[rival_soldier]
            self.board[rival_prev_pos] = 0
            self.board[rival_soldier_cell] = self.rival_index
            self.rival_pos[rival_soldier] = rival_soldier_cell
        if my_dead_pos != -1:
            self.board[my_dead_pos] = 0
            dead_soldier = int(np.where(self.player_pos == my_dead_pos)[0][0])
            self.player_pos[dead_soldier] = -2
        self.turn_count += 1

    ########## helper functions in class ##########
    # TODO: add here helper functions in class, if needed

    def check_won_game(self, player_idx):
        rival_pos = self.rival_pos if player_idx == 1 else self.player_pos
        dead = np.count_nonzero(rival_pos == -2)
        if dead > 6:
            return True
        for index, x in enumerate(rival_pos):
            if x > -1 and not self.check_if_blocked(x):
                return False
        if self.turn_count < 18:
            return False
        return True

    def enable_heavy_player(self):
        self.heavy_player = True

    def enable_light_player(self):
        self.light_player = True

    ########## helper functions for AlphaBeta algorithm ##########
    # TODO: add here the utility, succ, an
    def update_move(self, best_move):
        cell, my_soldier, rival_soldier_cell = best_move
        if self.turn_count < 18:
            self.board[cell] = self.player_index
            self.player_pos[my_soldier] = cell
        else:
            player_prev_pos = self.player_pos[my_soldier]
            self.board[player_prev_pos] = 0
            self.board[cell] = self.player_index
            self.player_pos[my_soldier] = cell
        if rival_soldier_cell != -1:
            self.board[rival_soldier_cell] = 0
            dead_soldier = int(np.where(self.rival_pos == rival_soldier_cell)[0][0])
            self.rival_pos[dead_soldier] = -2
        self.turn_count += 1

    def simple_stage_1_move(self) -> tuple:
        # cell = int(np.where(self.board == 0)[0][0])
        # soldier_that_moved = int(np.where(self.my_pos == -1)[0][0])
        cell, soldier_that_moved = self._stage_1_choose_cell_and_soldier_to_move()
        self.player_pos[soldier_that_moved] = cell
        self.board[cell] = 1
        rival_cell = -1 if not self.is_mill(cell) else self._make_mill_get_rival_cell()
        return cell, soldier_that_moved, rival_cell

    def _stage_1_choose_cell_and_soldier_to_move(self):
        cell = int(np.where(self.board == 0)[0][0])
        soldier_that_moved = int(np.where(self.player_pos == -1)[0][0])
        return cell, soldier_that_moved

    def _make_mill_get_rival_cell(self):
        rival_cell = self._choose_rival_cell_to_kill()
        rival_idx = np.where(self.rival_pos == rival_cell)[0][0]
        self.rival_pos[rival_idx] = -2
        self.board[rival_cell] = 0
        return rival_cell

    def _choose_rival_cell_to_kill(self):
        rival_cell = np.where(self.board == 2)[0][0]
        return rival_cell

    def simple_stage_2_move(self) -> tuple:
        cell, soldier_that_moved = -1, -1
        soldiers_on_board = np.where(self.board == 1)[0]
        for soldier_cell in soldiers_on_board:
            direction_list = self.directions(int(soldier_cell))
            for direction in direction_list:
                if self.board[direction] == 0:
                    cell = direction
                    soldier_that_moved = int(np.where(self.player_pos == soldier_cell)[0][0])
                    self._update_player_on_board(cell, self.player_pos[soldier_that_moved], soldier_that_moved)
                    rival_cell = -1 if not self.is_mill(cell) else self._make_mill_get_rival_cell()  # Check if mill
                    return cell, soldier_that_moved, rival_cell

    def _update_player_on_board(self, next_pos, prev_pos, soldier):
        # update position and board:
        self.board[next_pos] = 1
        self.board[prev_pos] = 0
        self.player_pos[soldier] = next_pos

    def enable_alpha_beta(self):
        self.AlphaBeta = True

    def switch_player_rival(self):
        tmp = self.player_pos
        self.player_pos = self.rival_pos
        self.rival_pos = tmp
        self.player_index = self.rival_index
        self.rival_index = 3 - self.player_index
        self.is_switched = 1 - self.is_switched

    def check_if_blocked(self, position, board=None):
        """
        Function to check if a player can make a mill in the next move.
        :param position: curren position
        :param board: np.array
        :param player: 1/2
        :return:
        """
        if board is None:
            board = self.board

        for i in utils.get_directions(position):
            if board[i] == 0:
                return False
        return True

    def is_double_mill(self, position):
        """
        Return True if a player has a mill on the given position
        :param position: 0-23
        :return:
        """
        if position < 0 or position > 23:
            return False
        p = int(self.board[position])

        # The player on that position
        if p != 0:
            # If there is some player on that position
            return self.check_double_mill(position, p, self.board)
        else:

            return False

    def check_double_mill(self, position, player, board):
        mill = [
            (self.is_player(player, 1, 2, board) and self.is_player(player, 3, 5, board)),
            (self.is_player(player, 0, 2, board) and self.is_player(player, 9, 17, board)),
            (self.is_player(player, 0, 1, board) and self.is_player(player, 4, 7, board)),
            (self.is_player(player, 0, 5, board) and self.is_player(player, 11, 19, board)),
            (self.is_player(player, 2, 7, board) and self.is_player(player, 12, 20, board)),
            (self.is_player(player, 0, 3, board) and self.is_player(player, 6, 7, board)),
            (self.is_player(player, 5, 7, board) and self.is_player(player, 14, 22, board)),
            (self.is_player(player, 2, 4, board) and self.is_player(player, 5, 6, board)),
            (self.is_player(player, 9, 10, board) and self.is_player(player, 11, 13, board)),
            (self.is_player(player, 8, 10, board) and self.is_player(player, 1, 17, board)),
            (self.is_player(player, 8, 9, board) and self.is_player(player, 12, 15, board)),
            (self.is_player(player, 3, 19, board) and self.is_player(player, 8, 13, board)),
            (self.is_player(player, 20, 4, board) and self.is_player(player, 10, 15, board)),
            (self.is_player(player, 8, 11, board) and self.is_player(player, 14, 15, board)),
            (self.is_player(player, 13, 15, board) and self.is_player(player, 6, 22, board)),
            (self.is_player(player, 13, 14, board) and self.is_player(player, 10, 12, board)),
            (self.is_player(player, 17, 18, board) and self.is_player(player, 19, 21, board)),
            (self.is_player(player, 1, 9, board) and self.is_player(player, 16, 18, board)),
            (self.is_player(player, 16, 17, board) and self.is_player(player, 20, 23, board)),
            (self.is_player(player, 16, 21, board) and self.is_player(player, 3, 11, board)),
            (self.is_player(player, 12, 4, board) and self.is_player(player, 18, 23, board)),
            (self.is_player(player, 16, 19, board) and self.is_player(player, 22, 23, board)),
            (self.is_player(player, 6, 14, board) and self.is_player(player, 21, 23, board)),
            (self.is_player(player, 18, 20, board) and self.is_player(player, 21, 22, board))
        ]
        return mill[position]

    def is_unblocked_mill(self, position, player, board=None):
        """
        Function to check if a player has an unblocked mill.
        :param position: curren position
        :param board: np.array
        :param player: 1/2
        :return:
        """
        if board is None:
            board = self.board

        blocked = [
            (board[0] == 0 and (self.is_player(player, 3, 5, board) and self.is_player(player, 1, 3, board) or \
                                self.is_player(player, 1, 2, board) and self.is_player(player, 1, 3, board))),

            (board[1] == 0 and (self.is_player(player, 0, 2, board) and self.is_player(player, 0, 9, board) or \
                                self.is_player(player, 9, 17, board) and self.is_player(player, 0, 2, board))),

            (board[2] == 0 and (self.is_player(player, 4, 7, board) and self.is_player(player, 1, 4, board) or \
                                self.is_player(player, 0, 1, board) and self.is_player(player, 1, 4, board))),

            (board[3] == 0 and (self.is_player(player, 0, 11, board) and self.is_player(player, 0, 5, board) or \
                                self.is_player(player, 11, 19, board) and self.is_player(player, 0, 5, board))),

            (board[4] == 0 and (self.is_player(player, 2, 7, board) and self.is_player(player, 2, 12, board) or \
                                self.is_player(player, 2, 7, board) and self.is_player(player, 12, 20, board))),

            (board[5] == 0 and (self.is_player(player, 6, 7, board) and self.is_player(player, 3, 6, board) or \
                                self.is_player(player, 0, 3, board) and self.is_player(player, 3, 6, board))),

            (board[6] == 0 and (self.is_player(player, 5, 7, board) and self.is_player(player, 5, 14, board) or \
                                self.is_player(player, 14, 22, board) and self.is_player(player, 5, 14, board))),

            (board[7] == 0 and (self.is_player(player, 5, 6, board) and self.is_player(player, 6, 4, board) or \
                                self.is_player(player, 2, 4, board) and self.is_player(player, 6, 4, board))),

            (board[8] == 0 and (self.is_player(player, 9, 10, board) and self.is_player(player, 10, 11, board) or \
                                self.is_player(player, 11, 13, board) and self.is_player(player, 10, 11, board))),

            (board[9] == 0 and (self.is_player(player, 1, 17, board) and self.is_player(player, 8, 10, board))),

            (board[10] == 0 and (self.is_player(player, 8, 9, board) and self.is_player(player, 9, 12, board) or \
                                 self.is_player(player, 12, 15, board) and self.is_player(player, 9, 12, board))),

            (board[11] == 0 and (self.is_player(player, 3, 19, board) and self.is_player(player, 8, 13, board))),

            (board[12] == 0 and (self.is_player(player, 4, 20, board) and self.is_player(player, 10, 15, board))),

            (board[13] == 0 and (self.is_player(player, 8, 11, board) and self.is_player(player, 11, 14, board) or \
                                 self.is_player(player, 14, 15, board) and self.is_player(player, 11, 14, board))),

            (board[14] == 0 and (self.is_player(player, 6, 22, board) and self.is_player(player, 13, 15, board))),

            (board[15] == 0 and (self.is_player(player, 13, 14, board) and self.is_player(player, 12, 14, board) or \
                                 self.is_player(player, 10, 12, board) and self.is_player(player, 12, 14, board))),

            (board[16] == 0 and (self.is_player(player, 17, 18, board) and self.is_player(player, 18, 19, board) or \
                                 self.is_player(player, 19, 21, board) and self.is_player(player, 18, 19, board))),

            (board[17] == 0 and (self.is_player(player, 16, 18, board) and self.is_player(player, 16, 9, board) or \
                                 self.is_player(player, 1, 9, board) and self.is_player(player, 16, 18, board))),

            (board[18] == 0 and (self.is_player(player, 16, 17, board) and self.is_player(player, 17, 20, board) or \
                                 self.is_player(player, 20, 23, board) and self.is_player(player, 17, 20, board))),

            (board[19] == 0 and (self.is_player(player, 16, 21, board) and self.is_player(player, 16, 11, board) or \
                                 self.is_player(player, 3, 11, board) and self.is_player(player, 16, 21, board))),

            (board[20] == 0 and (self.is_player(player, 18, 23, board) and self.is_player(player, 12, 18, board) or \
                                 self.is_player(player, 4, 12, board) and self.is_player(player, 18, 23, board))),

            (board[21] == 0 and (self.is_player(player, 16, 19, board) and self.is_player(player, 19, 22, board) or \
                                 self.is_player(player, 22, 23, board) and self.is_player(player, 19, 22, board))),

            (board[22] == 0 and (self.is_player(player, 21, 23, board) and self.is_player(player, 14, 21, board) or \
                                 self.is_player(player, 6, 14, board) and self.is_player(player, 21, 23, board))),

            (board[23] == 0 and (self.is_player(player, 18, 20, board) and self.is_player(player, 20, 22, board) or \
                                 self.is_player(player, 21, 22, board) and self.is_player(player, 20, 22, board)))
        ]

        return blocked[position]


def calculate_state_heuristic(state):
    player = state[0]
    if player.is_switched == 1:
        player.switch_player_rival()
    player_mill_num = 0
    rival_mill_num = 0
    player_incomplete_mills = 0
    rival_incomplete_mills = 0
    player_blocked_soldiers = 0
    rival_blocked_soldiers = 0
    rival_double_mill = 0
    player_double_mill = 0
    player_winning_config = 1 if player.check_won_game(player.player_index) else 0
    rival_winning_config = 1 if player.check_won_game(player.rival_index) else 0
    player_soldier = np.count_nonzero(player.player_pos > -1)
    rival_soldier = np.count_nonzero(player.rival_pos > -1)
    board = player.board
    for index, x in enumerate(board):
        cell = int(x)
        if cell == player.player_index:
            if player.is_mill(index):
                player_mill_num += 1 / 3
            if player.check_if_blocked(index, board):
                player_blocked_soldiers += 1
            if player.is_double_mill(index):
                player_double_mill += 1
        elif cell == player.rival_index:
            if player.is_mill(index):
                rival_mill_num += 1 / 3
            if player.is_double_mill(index):
                rival_double_mill += 1
            if player.check_if_blocked(index, board):
                rival_blocked_soldiers += 1
        elif cell == 0:
            if player.check_next_mill(index, player.player_index, board):
                player_incomplete_mills += 1
            if player.check_next_mill(index, player.player_index, board):
                rival_incomplete_mills += 1
    player_three_config = 1 if player_incomplete_mills >= 2 else 0
    rival_three_config = 1 if rival_incomplete_mills >= 2 else 0
    if player.turn_count < 18:
        return 10 * player_mill_num - 8 * rival_mill_num + \
               8 * player_incomplete_mills - 6 * rival_incomplete_mills + \
               15 * player_soldier - 15 * rival_soldier + \
               4 * player_three_config - 4 * rival_three_config + \
               4 * player_double_mill - 4 * rival_double_mill + \
               1000 * (player_winning_config - rival_winning_config)
    else:
        return 10 * player_mill_num - 10 * rival_mill_num + \
               25 * player_soldier - 25 * rival_soldier + \
               10 * player_incomplete_mills - 10 * rival_incomplete_mills + \
               4 * (rival_blocked_soldiers - player_blocked_soldiers) + \
               1000 * (player_winning_config - rival_winning_config) + \
               2 * (player_double_mill - rival_double_mill)


def calculate_simple_heuristic(state):
    player = state[0]
    if player.is_switched == 1:
        player.switch_player_rival()
    player_mill_num = 0
    rival_mill_num = 0
    board = player.board
    for index, x in enumerate(board):
        cell = int(x)
        if cell == player.player_index:
            if player.is_mill(index):
                player_mill_num += 1 / 3
        elif cell == player.rival_index:
            if player.is_mill(index):
                rival_mill_num += 1 / 3
    return player_mill_num-rival_mill_num


def succ(player, direction):
    # PHASE 1
    player = copy.deepcopy(player)
    if player.turn_count < 18:
        soldier_that_moved = int(np.where(player.player_pos == -1)[0][0])
        for cell, val in enumerate(player.board):
            if val == 0:
                player2 = copy.deepcopy(player)
                player2.player_pos[soldier_that_moved] = cell
                player2.board[cell] = player2.player_index
                player2.turn_count += 1
                if player2.is_mill(cell):
                    for index, to_kill in enumerate(player2.rival_pos):
                        if to_kill not in [-1, -2]:
                            player3 = copy.deepcopy(player2)
                            player3.board[to_kill] = 0
                            player3.rival_pos[index] = -2
                            player3.switch_player_rival()
                            yield player3, (cell, soldier_that_moved, to_kill)
                else:
                    player2.switch_player_rival()
                    yield player2, (cell, soldier_that_moved, -1)
    else:
        # PHASE 2
        for i, cell in enumerate(player.player_pos):
            if cell in [-1, -2]:
                continue
            directions = utils.get_directions(cell)
            for d in directions:
                if (d not in player.rival_pos) and (d not in player.player_pos):
                    player2 = copy.deepcopy(player)
                    player2.turn_count += 1
                    player2.board[cell] = 0
                    player2.board[d] = player2.player_index
                    player2.player_pos[i] = d
                    if player2.is_mill(d, player2.board):
                        for index, to_kill in enumerate(player2.rival_pos):
                            if to_kill not in [-1, -2]:
                                player3 = copy.deepcopy(player2)
                                player3.board[to_kill] = 0
                                player3.rival_pos[index] = -2
                                if (d, i, to_kill) == (2, 0, 8):
                                    print()
                                player3.switch_player_rival()
                                yield player3, (d, i, to_kill)
                    else:
                        player2.switch_player_rival()
                        yield player2, (d, i, -1)


def is_goal(state) -> bool:
    blocked = True
    dead = False
    exists_soldier = False
    player = state[0]
    dead1 = np.count_nonzero(player.player_pos == -2)
    dead2 = np.count_nonzero(player.rival_pos == -2)
    dead = True if dead1 > 6 else dead
    dead = True if dead2 > 6 else dead
    if np.count_nonzero(player.player_pos == -1) == 9 or np.count_nonzero(player.player_pos == -1) == 9:
        return False
    for cell in player.player_pos:
        if cell not in [-1, -2]:
            exists_soldier = True
            if not player.check_if_blocked(cell):
                blocked = False
    blocked = False if not exists_soldier else blocked
    blocked = False if player.turn_count < 18 else blocked
    return blocked or dead