import unittest

from main import (
    search_bfs,
    search_dfs,
    search_best_first,
    search_astar,
)


class SearchAlgorithmTests(unittest.TestCase):
    def setUp(self):
        self.rows = 5
        self.cols = 5
        self.walls = {(1, 1), (1, 2), (1, 3), (3, 1), (3, 2)}
        self.start = (0, 0)
        self.goal = (4, 4)

    def test_bfs_finds_path(self):
        order, path = search_bfs(self.rows, self.cols, self.walls, self.start, self.goal)
        self.assertTrue(order)
        self.assertEqual(path[0], self.start)
        self.assertEqual(path[-1], self.goal)

    def test_astar_path_length_matches_bfs(self):
        _, bfs_path = search_bfs(self.rows, self.cols, self.walls, self.start, self.goal)
        _, astar_path = search_astar(self.rows, self.cols, self.walls, self.start, self.goal)
        self.assertEqual(len(astar_path), len(bfs_path))

    def test_no_path(self):
        walls = {(0, 1), (1, 0)}
        _, path = search_bfs(self.rows, self.cols, walls, self.start, self.goal)
        self.assertEqual(path, [])

    def test_dfs_and_greedy_find_a_path_on_open_grid(self):
        for algorithm in (search_dfs, search_best_first, search_astar):
            _, path = algorithm(self.rows, self.cols, set(), self.start, self.goal)
            self.assertTrue(path)
            self.assertEqual(path[0], self.start)
            self.assertEqual(path[-1], self.goal)


if __name__ == "__main__":
    unittest.main()
