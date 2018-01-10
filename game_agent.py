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
    nMoves_active_player = len(game.get_legal_moves(player))
    nMoves_inactive_player = len(game.get_legal_moves(game.get_opponent(player)))
    return 1.0 * nMoves_active_player

def custom_score_2(game, player):
    nMoves_active_player = len(game.get_legal_moves(player))
    nMoves_inactive_player = len(game.get_legal_moves(game.get_opponent(player)))
    return 1.0 * nMoves_active_player - nMoves_inactive_player

def custom_score_3(game, player):
    nMoves_active_player = len(game.get_legal_moves(player))
    nMoves_inactive_player = len(game.get_legal_moves(game.get_opponent(player)))
    if nMoves_inactive_player == 0:
        return nMoves_active_player
    else:
        return nMoves_active_player / nMoves_inactive_player

class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
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
        #print("in MinimaxPlayer.get_move()")

        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)
        try:
            return self.minimax(game, self.search_depth)
        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed
        return best_move

    def minimax(self, game, depth):
        #print("in MinimaxPlayer.minimax()")
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        best_score = float("-inf")
        best_move = None
        for aMove in game.get_legal_moves():
            vBoard = self.min_value(game.forecast_move(aMove), depth-1)
            if vBoard > best_score:
                best_score = vBoard
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
        vBoard = float("-inf")
        for aMove in game.get_legal_moves():
            vBoard = max(vBoard, self.min_value(game.forecast_move(aMove), depth-1))
        return vBoard

    def min_value(self, game, depth):
        #print("in MinimaxPlayer.min_value()")
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        if depth == 0 or self.terminal_test(game):
            return self.score(game, self)
        vBoard = float("inf")
        for aMove in game.get_legal_moves():
            vBoard = min(vBoard, self.max_value(game.forecast_move(aMove), depth-1))
        return vBoard

class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        #print("in AlphaBetaPlayer.get_move()")
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)
        try:
            depth = 1
            while True:
                if self.time_left() < self.TIMER_THRESHOLD:
                    raise SearchTimeout()
                best_move = self.alphabeta(game, depth)
                depth += 1

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed
        return best_move

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        #print("in AlphaBetaPlayer.alphabeta()")
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        """
        # print("in AlphaBetaPlayer.alphabeta()")
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        best_score = float("-inf")
        best_move = None
        for aMove in game.get_legal_moves():
            vBoard = self.min_value(game.forecast_move(aMove), depth-1, alpha, beta)
            if vBoard > best_score:
                best_score = vBoard
                best_move = aMove
            alpha = max(alpha, vBoard)
        return best_move

    def terminal_test(self, game):
        #print("in MinimaxPlayer.terminal_test()")
        return not bool(game.get_legal_moves())

    def max_value(self, game, depth, alpha, beta):
        #print("in MinimaxPlayer.max_value()")
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        if depth == 0 or self.terminal_test(game):
            return self.score(game, self)
        vBoard = float("-inf")
        for aMove in game.get_legal_moves():
            vBoard = max(vBoard, self.min_value(game.forecast_move(aMove), depth-1, alpha, beta))
            if vBoard >= beta:
                return vBoard
            alpha = max(alpha, vBoard)
        return vBoard

    def min_value(self, game, depth, alpha, beta):
        #print("in MinimaxPlayer.min_value()")
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        if depth == 0 or self.terminal_test(game):
            return self.score(game, self)
        vBoard = float("inf")
        for aMove in game.get_legal_moves():
            vBoard = min(vBoard, self.max_value(game.forecast_move(aMove), depth-1, alpha, beta))
            if vBoard <= alpha:
                return vBoard
            beta = min(beta, vBoard)
        return vBoard
