"""Implement your own custom search agent using any combination of techniques
you choose.  This agent will compete against other students (and past
champions) in a tournament.

         COMPLETING AND SUBMITTING A COMPETITION AGENT IS OPTIONAL
"""
import random


class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

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

    if center in moves_my_player:
        factor_my_player += 100
    elif center in moves_opp_player:
        factor_opp_player += 100

    return factor_my_player - factor_opp_player

class CustomPlayer:
    """Game-playing agent to use in the optional player vs player Isolation
    competition.

    You must at least implement the get_move() method and a search function
    to complete this class, but you may use any of the techniques discussed
    in lecture or elsewhere on the web -- opening books, MCTS, etc.

    **************************************************************************
          THIS CLASS IS OPTIONAL -- IT IS ONLY USED IN THE ISOLATION PvP
        COMPETITION.  IT IS NOT REQUIRED FOR THE ISOLATION PROJECT REVIEW.
    **************************************************************************

    Parameters
    ----------
    data : string
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted.  Note that
        the PvP competition uses more accurate timers that are not cross-
        platform compatible, so a limit of 1ms (vs 10ms for the other classes)
        is generally sufficient.
    """

    def __init__(self, data=None, timeout=1.):
        self.score = custom_score
        self.time_left = None
        self.TIMER_THRESHOLD = timeout

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
        return not bool(game.get_legal_moves())

    def max_value(self, game, depth, alpha, beta):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        if depth == 0 or self.terminal_test(game):
            return self.score(game, self)
        aScore = float("-inf")
        for aMove in game.get_legal_moves():
            aScore = max(aScore, self.min_value(game.forecast_move(aMove), depth-1, alpha, beta))
            if aScore >= beta:
                return aScore
            alpha = max(alpha, aScore)
        return aScore

    def min_value(self, game, depth, alpha, beta):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        if depth == 0 or self.terminal_test(game):
            return self.score(game, self)
        aScore = float("inf")
        for aMove in game.get_legal_moves():
            aScore = min(aScore, self.max_value(game.forecast_move(aMove), depth-1, alpha, beta))
            if aScore <= alpha:
                return aScore
            beta = min(beta, aScore)
        return aScore
