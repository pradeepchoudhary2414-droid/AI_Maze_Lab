# AI Maze Lab — Intelligent Pathfinding Game

A desktop mini-project for an AI Lab syllabus, built with Python and Pygame.

## Features
- Breadth First Search (BFS)
- Depth First Search (DFS)
- Greedy Best-First Search
- A* Search with Manhattan distance
- Animated cell exploration and final path
- Random obstacle maze generation
- Edit walls, move start/goal, pause/resume and adjust speed
- Live visited-cell and path-length metrics

## Requirements
- Python 3.10+
- pygame

## Run
```bash
python -m pip install -r requirements.txt
python main.py
```

## Controls
- Left click a cell: add/remove a wall
- Shift + left click: move start
- Ctrl + left click: move goal
- Space: run search / resume
- R: generate a new random maze
- C: clear walls
- 1: BFS, 2: DFS, 3: Best-First, 4: A*
- Up/Down are not used; use `+` / `-` to change animation speed
- Escape: quit

Right-click a wall cell to remove it quickly.

## Algorithm notes
- BFS finds a shortest path when every move has equal cost.
- DFS is not guaranteed to find a shortest path.
- Greedy Best-First prioritizes Manhattan distance to the goal; it is not guaranteed to find a shortest path.
- A* uses `f(n) = g(n) + h(n)`. With unit-cost four-direction movement and Manhattan distance, it finds a shortest path.

The execution-time metric measures search computation only; animation time is not included.

## Project limitations / future extensions
- Maze generation uses random obstacles rather than a perfect-maze generator.
- Optional extensions: save maze layouts, compare algorithms side-by-side, add sound effects, and export benchmark results.
