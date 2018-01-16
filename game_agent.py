"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random
import math

class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass

def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    moves_my_player = game.get_legal_moves(player)
    moves_opp_player = game.get_legal_moves(game.get_opponent(player))

    nMoves_my_player = len(moves_my_player)
    nMoves_opp_player = len(moves_opp_player)

    return float(nMoves_my_player - nMoves_opp_player)

def custom_score_2(game, player):
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    moves_my_player = game.get_legal_moves(player)
    moves_opp_player = game.get_legal_moves(game.get_opponent(player))

    nMoves_my_player = len(moves_my_player)
    nMoves_opp_player = len(moves_opp_player)

    factor_my_player = 0.
    factor_opp_player = 0.

    board_corners = [(0,0),(0,6),(6,0),(6,6)]
    three_moves = [(0,1),(1,0),(0,5),(1,6),(5,0),(6,1),(5,6),(6,5)]
    four_moves = [(0,2),(0,3),(0,4),(1,1),(1,5),(2,0),(3,0),(4,0),(2,6),(3,6),(4,6),(5,1),(5,5),(6,2),(6,3),(6,4)]
    six_moves = [(1,2),(1,3),(1,4),(2,1),(3,1),(4,1),(2,5),(3,5),(4,5),(5,2),(5,3),(5,4)]
    center = (3, 3)

    for aMove in moves_my_player:
        if aMove in board_corners:
            factor_my_player += 2
        elif aMove in three_moves:
            factor_my_player += 3
        elif aMove in four_moves:
            factor_my_player += 4
        elif aMove in six_moves:
            factor_my_player += 6
        else:
            factor_my_player += 8

    for aMove in moves_opp_player:
        if aMove in board_corners:
            factor_opp_player += 2
        elif aMove in three_moves:
            factor_opp_player += 3
        elif aMove in four_moves:
            factor_opp_player += 4
        elif aMove in six_moves:
            factor_opp_player += 6
        else:
            factor_opp_player += 8

    # if center in moves_my_player:
    #     factor_my_player += 10
    # elif center in moves_opp_player:
    #     factor_opp_player += 10

    return factor_my_player - factor_opp_player

def custom_score_3(game, player):
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    moves_my_player = game.get_legal_moves(player)
    moves_opp_player = game.get_legal_moves(game.get_opponent(player))

    nMoves_my_player = len(moves_my_player)
    nMoves_opp_player = len(moves_opp_player)

    factor_my_player = 0.
    factor_opp_player = 0.

    board_corners = [(0,0),(0,6),(6,0),(6,6)]
    three_moves = [(0,1),(1,0),(0,5),(1,6),(5,0),(6,1),(5,6),(6,5)]
    four_moves = [(0,2),(0,3),(0,4),(1,1),(1,5),(2,0),(3,0),(4,0),(2,6),(3,6),(4,6),(5,1),(5,5),(6,2),(6,3),(6,4)]
    six_moves = [(1,2),(1,3),(1,4),(2,1),(3,1),(4,1),(2,5),(3,5),(4,5),(5,2),(5,3),(5,4)]
    center = (3, 3)

    for aMove in moves_my_player:
        if aMove in board_corners:
            factor_my_player += 4
        elif aMove in three_moves:
            factor_my_player += 9
        elif aMove in four_moves:
            factor_my_player += 16
        elif aMove in six_moves:
            factor_my_player += 36
        else:
            factor_my_player += 64

    for aMove in moves_opp_player:
        if aMove in board_corners:
            factor_opp_player += 4
        elif aMove in three_moves:
            factor_opp_player += 9
        elif aMove in four_moves:
            factor_opp_player += 16
        elif aMove in six_moves:
            factor_opp_player += 36
        else:
            factor_opp_player += 64

    # if center in moves_my_player:
    #     factor_my_player += 100
    # elif center in moves_opp_player:
    #     factor_opp_player += 100

    return factor_my_player - factor_opp_player

class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=15.):
        #print("in IsolationPlayer.__init__()")
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout

class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        if self.time_left() <= 0:
            return best_move

        try:
            return self.minimax(game, self.search_depth)
        except SearchTimeout:
            pass
        return best_move

    def minimax(self, game, depth):
        #print("in MinimaxPlayer.minimax()")
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        best_score = float("-inf")
        best_move = None

        if len(game.get_legal_moves()) > 0:
            best_move = game.get_legal_moves()[0]

        for aMove in game.get_legal_moves():
            aScore = self.min_value(game.forecast_move(aMove), depth-1)
            if aScore > best_score:
                best_score = aScore
                best_move = aMove
        return best_move

    def terminal_test(self, game):
        #print("in MinimaxPlayer.terminal_test()")
        return not bool(game.get_legal_moves())

    def max_value(self, game, depth):
        #print("in MinimaxPlayer.max_value()")
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        if depth == 0 or self.terminal_test(game):
            return self.score(game, self)
        aScore = float("-inf")
        for aMove in game.get_legal_moves():
            aScore = max(aScore, self.min_value(game.forecast_move(aMove), depth-1))
        return aScore

    def min_value(self, game, depth):
        #print("in MinimaxPlayer.min_value()")
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        if depth == 0 or self.terminal_test(game):
            return self.score(game, self)
        aScore = float("inf")
        for aMove in game.get_legal_moves():
            aScore = min(aScore, self.max_value(game.forecast_move(aMove), depth-1))
        return aScore

class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        #print("AB_get_move()")
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        if self.time_left() <= 0:
            return best_move

        try:
            depth = 1
            while True:
                if self.time_left() < self.TIMER_THRESHOLD:
                    raise SearchTimeout()
                best_move = self.alphabeta(game, depth)
                depth += 1
        except SearchTimeout:
            return best_move
        return best_move

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        #print("AB_alphabeta()")
        best_score = float("-inf")
        best_move = None

        # print("in AlphaBetaPlayer.alphabeta()")
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        if len(game.get_legal_moves()) > 0:
            best_move = game.get_legal_moves()[0]

        for aMove in game.get_legal_moves():
            aScore = self.min_value(game.forecast_move(aMove), depth-1, alpha, beta)
            if aScore > best_score:
                best_score = aScore
                best_move = aMove
            alpha = max(alpha, aScore)
        return best_move

    def terminal_test(self, game):
        #print("in MinimaxPlayer.terminal_test()")
        return not bool(game.get_legal_moves())

    def max_value(self, game, depth, alpha, beta):
        #print("in AB_Max()")
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        if depth == 0 or self.terminal_test(game):
            #print("AB_Max: Cutoff. depth=", depth, "terminate=", self.terminal_test(game))
            #print("\tScore=", self.score(game, self))
            #game.to_string()
            return self.score(game, self)
        aScore = float("-inf")
        for aMove in game.get_legal_moves():
            aScore = max(aScore, self.min_value(game.forecast_move(aMove), depth-1, alpha, beta))
            if aScore >= beta:
                return aScore
            alpha = max(alpha, aScore)
        return aScore

    def min_value(self, game, depth, alpha, beta):
        #print("in AB_Min()")
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        if depth == 0 or self.terminal_test(game):
            #print("AB_Min: Cutoff. depth=", depth, "terminate=", self.terminal_test(game))
            #print("\tScore=", self.score(game, self))
            #game.to_string()
            return self.score(game, self)
        aScore = float("inf")
        for aMove in game.get_legal_moves():
            aScore = min(aScore, self.max_value(game.forecast_move(aMove), depth-1, alpha, beta))
            if aScore <= alpha:
                return aScore
            beta = min(beta, aScore)
        return aScore
