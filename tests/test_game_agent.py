import unittest
import isolation
import game_agent

from importlib import reload

class IsolationTest(unittest.TestCase):
    """Unit tests for isolation agents"""

    def setUp(self):
        print("in IsolationTest.setUp()")
        reload(game_agent)
        self.player1 = game_agent.AlphaBetaPlayer()
        self.player2 = game_agent.AlphaBetaPlayer()
        self.game = isolation.Board(self.player1, self.player2)
        #self.test_example()

    def test_example(self):
        print("in IsolationTest.test_example()")
        self.game.play(2000)
        #self.fail("Hello, World!")

if __name__ == '__main__':
    unittest.main()
