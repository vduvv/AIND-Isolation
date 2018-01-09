import random

class SearchTimeout(Exception):
    pass


def custom_score(game, player):
    return -1

def custom_score_2(game, player):
    return -1


def custom_score_3(game, player):
    return -1

class IsolationPlayer:
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout

class MinimaxPlayer(IsolationPlayer):
    def get_move(self, game, time_left):
        self.time_left = time_left
        best_move = (-1, -1)
        try:
            return self.minimax(game, self.search_depth)
        except SearchTimeout:
            pass
        return best_move

    def minimax(self, game, depth):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        self.min_value(game)

    def terminal_test(game):
        return not bool(gameState.get_legal_moves())  # by Assumption 1

    def min_value(game):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        if terminal_test(game):
            return self.score(game, self)
        v = float("inf")
        for m in game.get_legal_moves():
            v = min(v, self.max_value(game.forecast_move(m)))
        return v

    def max_value(game):
        if self.time_left() < self.TIMER_THRESHOLD raise SearchTimeout()
        if terminal_test(game):
            return -1  # by assumption 2
        v = float("-inf")
        for m in game.get_legal_moves():
            v = max(v, self.min_value(game.forecast_move(m)))
        return v

class AlphaBetaPlayer(IsolationPlayer):
    def get_move(self, game, time_left):
        self.time_left = time_left

        # TODO: finish this function!
        raise NotImplementedError

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # TODO: finish this function!
        raise NotImplementedError
