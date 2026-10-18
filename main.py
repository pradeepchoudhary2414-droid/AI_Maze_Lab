"""
AI Maze Lab - Interactive Pathfinding Game
Python 3.10+ | Pygame

Controls:
  Left click empty cell/wall: toggle wall
  Shift + left click: move start
  Ctrl + left click: move goal
  Space: run / pause / resume
  R: generate a new random maze
  C: clear all walls
  1-4: select BFS / DFS / Greedy Best-First / A*
  Up/Down: adjust animation speed
  Esc: quit
"""

from __future__ import annotations

import heapq
import random
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable

import pygame


# ------------------------------ Configuration ------------------------------

WIDTH, HEIGHT = 1120, 780
FPS = 60
ROWS, COLS = 25, 25
CELL = 22
GRID_X, GRID_Y = 30, 118
GRID_W, GRID_H = COLS * CELL, ROWS * CELL
PANEL_X = GRID_X + GRID_W + 28
PANEL_W = WIDTH - PANEL_X - 28

BG = (11, 16, 31)
PANEL = (19, 27, 47)
PANEL_2 = (25, 35, 59)
BORDER = (47, 62, 91)
TEXT = (231, 239, 255)
MUTED = (142, 159, 190)
CYAN = (62, 220, 230)
PURPLE = (166, 119, 255)
GREEN = (73, 225, 157)
RED = (255, 100, 124)
YELLOW = (255, 204, 102)
WALL = (61, 76, 105)
GRID_LINE = (26, 36, 58)
VISITED = (40, 111, 158)
VISITED_2 = (49, 143, 175)
PATH = (255, 204, 102)
WHITE = (255, 255, 255)

ALGORITHMS = ["BFS", "DFS", "Best-First", "A*"]
ALGORITHM_DESCRIPTIONS = {
    "BFS": "Explores evenly by distance. Finds a shortest path on an unweighted grid.",
    "DFS": "Explores deep along one branch. Does not guarantee the shortest path.",
    "Best-First": "Prioritizes cells that look closest to the goal using a heuristic.",
    "A*": "Combines path cost and heuristic; finds a shortest path with this grid heuristic.",
}


# ------------------------------ Search algorithms ---------------------------

def neighbors(node: tuple[int, int], rows: int, cols: int):
    r, c = node
    for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
        if 0 <= nr < rows and 0 <= nc < cols:
            yield (nr, nc)


def manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def reconstruct_path(parent: dict, start: tuple[int, int], goal: tuple[int, int]) -> list:
    if goal != start and goal not in parent:
        return []
    path = [goal]
    current = goal
    while current != start:
        current = parent[current]
        path.append(current)
    path.reverse()
    return path


def search_bfs(rows, cols, walls, start, goal):
    queue = deque([start])
    seen = {start}
    parent = {}
    order = []
    while queue:
        current = queue.popleft()
        order.append(current)
        if current == goal:
            break
        for nxt in neighbors(current, rows, cols):
            if nxt not in walls and nxt not in seen:
                seen.add(nxt)
                parent[nxt] = current
                queue.append(nxt)
    return order, reconstruct_path(parent, start, goal) if goal in seen else []


def search_dfs(rows, cols, walls, start, goal):
    stack = [start]
    seen = {start}
    parent = {}
    order = []
    while stack:
        current = stack.pop()
        order.append(current)
        if current == goal:
            break
        # Reverse order keeps the visual exploration reasonably consistent.
        for nxt in reversed(list(neighbors(current, rows, cols))):
            if nxt not in walls and nxt not in seen:
                seen.add(nxt)
                parent[nxt] = current
                stack.append(nxt)
    return order, reconstruct_path(parent, start, goal) if goal in seen else []


def search_best_first(rows, cols, walls, start, goal):
    counter = 0
    heap = [(manhattan(start, goal), counter, start)]
    seen = {start}
    parent = {}
    order = []
    while heap:
        _, _, current = heapq.heappop(heap)
        order.append(current)
        if current == goal:
            break
        for nxt in neighbors(current, rows, cols):
            if nxt not in walls and nxt not in seen:
                seen.add(nxt)
                parent[nxt] = current
                counter += 1
                heapq.heappush(heap, (manhattan(nxt, goal), counter, nxt))
    return order, reconstruct_path(parent, start, goal) if goal in seen else []


def search_astar(rows, cols, walls, start, goal):
    counter = 0
    heap = [(manhattan(start, goal), 0, counter, start)]
    g_score = {start: 0}
    parent = {}
    closed = set()
    order = []
    while heap:
        _, g, _, current = heapq.heappop(heap)
        if current in closed:
            continue
        closed.add(current)
        order.append(current)
        if current == goal:
            break
        for nxt in neighbors(current, rows, cols):
            if nxt in walls or nxt in closed:
                continue
            tentative = g + 1
            if tentative < g_score.get(nxt, float("inf")):
                g_score[nxt] = tentative
                parent[nxt] = current
                counter += 1
                f_score = tentative + manhattan(nxt, goal)
                heapq.heappush(heap, (f_score, tentative, counter, nxt))
    return order, reconstruct_path(parent, start, goal) if goal in closed else []


SEARCHERS: dict[str, Callable] = {
    "BFS": search_bfs,
    "DFS": search_dfs,
    "Best-First": search_best_first,
    "A*": search_astar,
}


# -------------------------------- UI helpers --------------------------------

@dataclass
class Button:
    rect: pygame.Rect
    label: str
    action: str
    selected: bool = False
    accent: tuple = CYAN

    def draw(self, screen, font, mouse_pos):
        hovered = self.rect.collidepoint(mouse_pos)
        if self.selected:
            fill = (35, 67, 94)
            outline = self.accent
        elif hovered:
            fill = (35, 48, 75)
            outline = (86, 113, 157)
        else:
            fill = PANEL_2
            outline = BORDER
        pygame.draw.rect(screen, fill, self.rect, border_radius=9)
        pygame.draw.rect(screen, outline, self.rect, width=1, border_radius=9)
        label = font.render(self.label, True, TEXT if (self.selected or hovered) else MUTED)
        screen.blit(label, label.get_rect(center=self.rect.center))


def draw_text(screen, font, text, pos, color=TEXT):
    screen.blit(font.render(str(text), True, color), pos)


def rounded_panel(screen, rect, fill=PANEL, outline=BORDER, radius=16):
    pygame.draw.rect(screen, fill, rect, border_radius=radius)
    pygame.draw.rect(screen, outline, rect, width=1, border_radius=radius)


# -------------------------------- Main game ---------------------------------

class AIMazeLab:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("AI Maze Lab | Intelligent Pathfinding")
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("segoeui", 16)
        self.small = pygame.font.SysFont("segoeui", 13)
        self.medium = pygame.font.SysFont("segoeui", 20, bold=True)
        self.large = pygame.font.SysFont("segoeui", 30, bold=True)
        self.running = True

        self.start = (1, 1)
        self.goal = (ROWS - 2, COLS - 2)
        self.walls: set[tuple[int, int]] = set()
        self.algorithm = "A*"
        self.speed = 85  # animation steps per second
        self.state = "idle"  # idle, searching, path, paused, done, no_path
        self.visited_order = []
        self.path = []
        self.visited_shown = 0
        self.path_shown = 0
        self.search_elapsed = 0.0
        self.search_started = 0.0
        self.last_step_time = 0.0
        self.nodes_count = 0
        self.result_message = "Choose an algorithm and press RUN."
        self.buttons: list[Button] = []
        self._build_buttons()
        self.generate_maze()

    def _build_buttons(self):
        x = PANEL_X + 18
        w = (PANEL_W - 46) // 2
        y = 194
        self.buttons = [
            Button(pygame.Rect(x, y, w, 38), "BFS  [1]", "BFS", accent=CYAN),
            Button(pygame.Rect(x + w + 10, y, w, 38), "DFS  [2]", "DFS", accent=PURPLE),
            Button(pygame.Rect(x, y + 48, w, 38), "Best-First [3]", "Best-First", accent=YELLOW),
            Button(pygame.Rect(x + w + 10, y + 48, w, 38), "A*  [4]", "A*", accent=GREEN),
            Button(pygame.Rect(x, y + 110, PANEL_W - 36, 42), "RUN SEARCH  [SPACE]", "run", accent=CYAN),
            Button(pygame.Rect(x, y + 160, w, 38), "NEW MAZE [R]", "maze", accent=PURPLE),
            Button(pygame.Rect(x + w + 10, y + 160, w, 38), "CLEAR [C]", "clear", accent=RED),
            Button(pygame.Rect(x, y + 208, w, 38), "PAUSE / RESUME", "pause", accent=YELLOW),
            Button(pygame.Rect(x + w + 10, y + 208, w, 38), "RESET SEARCH", "reset", accent=CYAN),
            Button(pygame.Rect(x, y + 256, w, 38), "SLOWER  [-]", "slower", accent=PURPLE),
            Button(pygame.Rect(x + w + 10, y + 256, w, 38), "FASTER  [+]", "faster", accent=GREEN),
        ]

    def reset_search(self):
        self.state = "idle"
        self.visited_order = []
        self.path = []
        self.visited_shown = 0
        self.path_shown = 0
        self.search_elapsed = 0.0
        self.nodes_count = 0
        self.result_message = "Choose an algorithm and press RUN."

    def generate_maze(self):
        # Random obstacles, retry until at least one path exists.
        self.reset_search()
        for _ in range(100):
            walls = set()
            for r in range(ROWS):
                for c in range(COLS):
                    if (r, c) not in (self.start, self.goal) and random.random() < 0.27:
                        walls.add((r, c))
            _, path = search_bfs(ROWS, COLS, walls, self.start, self.goal)
            if path:
                self.walls = walls
                self.result_message = "New maze ready. Pick an algorithm."
                return
        self.walls = set()
        self.result_message = "New maze ready. Pick an algorithm."

    def clear_walls(self):
        self.walls.clear()
        self.reset_search()
        self.result_message = "Walls cleared. Add your own obstacles or run search."

    def run_search(self):
        if self.state in ("searching", "path"):
            return
        t0 = time.perf_counter()
        self.visited_order, self.path = SEARCHERS[self.algorithm](
            ROWS, COLS, self.walls, self.start, self.goal
        )
        self.search_elapsed = time.perf_counter() - t0
        self.search_started = time.perf_counter()
        self.last_step_time = 0
        self.visited_shown = 0
        self.path_shown = 0
        self.nodes_count = len(self.visited_order)
        if not self.visited_order or not self.path:
            self.state = "searching" if self.visited_order else "no_path"
            self.result_message = "No path found. Try clearing or changing walls."
            if self.state == "searching":
                self.result_message = "Exploring the maze..."
        else:
            self.state = "searching"
            self.result_message = f"{self.algorithm} is exploring the maze..."

    def handle_action(self, action):
        if action in ALGORITHMS:
            self.algorithm = action
            self.reset_search()
        elif action == "run":
            if self.state == "paused":
                self.state = "searching"
            else:
                self.run_search()
        elif action == "maze":
            self.generate_maze()
        elif action == "clear":
            self.clear_walls()
        elif action == "pause":
            if self.state in ("searching", "path"):
                self.state = "paused"
            elif self.state == "paused":
                self.state = "searching"
        elif action == "reset":
            self.reset_search()
        elif action == "slower":
            self.speed = max(5, self.speed - 20)
        elif action == "faster":
            self.speed = min(300, self.speed + 20)

    def update_animation(self, dt):
        if self.state not in ("searching", "path"):
            return
        self.last_step_time += dt
        step_interval = 1.0 / self.speed
        # Cap steps per frame to keep the UI responsive.
        steps = 0
        while self.last_step_time >= step_interval and steps < 8:
            self.last_step_time -= step_interval
            steps += 1
            if self.state == "searching":
                if self.visited_shown < len(self.visited_order):
                    self.visited_shown += 1
                if self.visited_shown >= len(self.visited_order):
                    if self.path:
                        self.state = "path"
                        self.result_message = "Search complete. Drawing final path..."
                    else:
                        self.state = "no_path"
                        self.result_message = "No path found. Try changing the walls."
                        break
            elif self.state == "path":
                if self.path_shown < len(self.path):
                    self.path_shown += 1
                if self.path_shown >= len(self.path):
                    self.state = "done"
                    self.result_message = f"Path found using {self.algorithm}!"

    def grid_cell_at(self, pos):
        x, y = pos
        if GRID_X <= x < GRID_X + GRID_W and GRID_Y <= y < GRID_Y + GRID_H:
            c = (x - GRID_X) // CELL
            r = (y - GRID_Y) // CELL
            return (int(r), int(c))
        return None

    def handle_grid_click(self, cell, mods):
        if cell is None:
            return
        if self.state in ("searching", "path"):
            return
        r, c = cell
        if mods & pygame.KMOD_SHIFT:
            if cell != self.goal and cell not in self.walls:
                self.start = cell
                self.reset_search()
        elif mods & pygame.KMOD_CTRL:
            if cell != self.start and cell not in self.walls:
                self.goal = cell
                self.reset_search()
        else:
            if cell in (self.start, self.goal):
                return
            if cell in self.walls:
                self.walls.remove(cell)
            else:
                self.walls.add(cell)
            self.reset_search()

    def draw_grid(self):
        visible_visited = set(self.visited_order[:self.visited_shown])
        visible_path = set(self.path[:self.path_shown])
        for r in range(ROWS):
            for c in range(COLS):
                cell = (r, c)
                rect = pygame.Rect(GRID_X + c * CELL, GRID_Y + r * CELL, CELL - 1, CELL - 1)
                color = (20, 29, 49)
                if cell in visible_visited:
                    color = VISITED_2 if (r + c) % 2 else VISITED
                if cell in self.walls:
                    color = WALL
                if cell in visible_path:
                    color = PATH
                if cell == self.start:
                    color = GREEN
                if cell == self.goal:
                    color = RED
                pygame.draw.rect(self.screen, color, rect, border_radius=3)

        # Subtle outer border around the board.
        pygame.draw.rect(self.screen, BORDER, (GRID_X - 2, GRID_Y - 2, GRID_W + 3, GRID_H + 3), 1, border_radius=7)

    def draw_header(self):
        draw_text(self.screen, self.large, "AI MAZE LAB", (30, 25), TEXT)
        draw_text(self.screen, self.font, "INTELLIGENT PATHFINDING SIMULATOR", (32, 65), CYAN)
        # Status pill
        status = self.state.upper().replace("_", " ")
        pill = pygame.Rect(WIDTH - 220, 31, 190, 34)
        pygame.draw.rect(self.screen, PANEL_2, pill, border_radius=17)
        pygame.draw.circle(self.screen, GREEN if self.state == "done" else CYAN, (pill.x + 17, pill.centery), 5)
        draw_text(self.screen, self.small, status, (pill.x + 30, pill.y + 10), TEXT)

    def draw_panel(self):
        panel_rect = pygame.Rect(PANEL_X, 105, PANEL_W, 625)
        rounded_panel(self.screen, panel_rect)
        x = PANEL_X + 18
        draw_text(self.screen, self.medium, "CONTROL CENTER", (x, 123), TEXT)
        draw_text(self.screen, self.small, "SELECT SEARCH ALGORITHM", (x, 163), MUTED)

        for b in self.buttons:
            b.selected = b.action == self.algorithm
            if b.action == "run":
                b.selected = False
            b.draw(self.screen, self.small, pygame.mouse.get_pos())

        # Explanation card
        card_y = 500
        card = pygame.Rect(x, card_y, PANEL_W - 36, 91)
        rounded_panel(self.screen, card, fill=(15, 22, 39), outline=BORDER, radius=11)
        draw_text(self.screen, self.small, f"ABOUT {self.algorithm.upper()}", (x + 12, card_y + 10), CYAN)
        desc = ALGORITHM_DESCRIPTIONS[self.algorithm]
        words = desc.split()
        lines, line = [], ""
        for word in words:
            test = (line + " " + word).strip()
            if self.small.size(test)[0] > card.width - 24:
                lines.append(line)
                line = word
            else:
                line = test
        if line:
            lines.append(line)
        for i, line in enumerate(lines[:3]):
            draw_text(self.screen, self.small, line, (x + 12, card_y + 34 + i * 17), MUTED)

        # Metrics
        metrics_y = 596
        draw_text(self.screen, self.small, "LIVE METRICS", (x, metrics_y), MUTED)
        metric_w = (PANEL_W - 46) // 2
        metric_rects = [
            pygame.Rect(x, metrics_y + 24, metric_w, 54),
            pygame.Rect(x + metric_w + 10, metrics_y + 24, metric_w, 54),
        ]
        values = [str(self.visited_shown), str(len(self.path) if self.path else 0)]
        labels = ["CELLS VISITED", "PATH LENGTH"]
        for rect, value, label in zip(metric_rects, values, labels):
            rounded_panel(self.screen, rect, fill=PANEL_2, outline=BORDER, radius=9)
            draw_text(self.screen, self.medium, value, (rect.x + 10, rect.y + 6), TEXT)
            draw_text(self.screen, self.small, label, (rect.x + 10, rect.y + 32), MUTED)

        elapsed_ms = self.search_elapsed * 1000
        draw_text(self.screen, self.small, f"Search compute time: {elapsed_ms:.3f} ms", (x, 674), MUTED)
        draw_text(self.screen, self.small, f"Animation speed: {self.speed} cells/sec", (x, 694), MUTED)

    def draw_footer(self):
        footer_y = 744
        text = "LEFT CLICK: wall  |  SHIFT+CLICK: start  |  CTRL+CLICK: goal  |  SPACE: run/pause  |  R: new maze"
        draw_text(self.screen, self.small, text, (30, footer_y), MUTED)
        # Status line below grid
        draw_text(self.screen, self.small, self.result_message[:78], (30, 686), CYAN if self.state == "done" else MUTED)

    def draw(self):
        self.screen.fill(BG)
        self.draw_header()
        draw_text(self.screen, self.medium, "MAZE CANVAS", (GRID_X, 88), TEXT)
        draw_text(self.screen, self.small, f"{ROWS} x {COLS} GRID", (GRID_X + 150, 94), MUTED)
        self.draw_grid()
        self.draw_panel()
        self.draw_footer()
        pygame.display.flip()

    def handle_event(self, event):
        if event.type == pygame.QUIT:
            self.running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.running = False
            elif event.key == pygame.K_SPACE:
                self.handle_action("run")
            elif event.key == pygame.K_r:
                self.handle_action("maze")
            elif event.key == pygame.K_c:
                self.handle_action("clear")
            elif event.key == pygame.K_1:
                self.handle_action("BFS")
            elif event.key == pygame.K_2:
                self.handle_action("DFS")
            elif event.key == pygame.K_3:
                self.handle_action("Best-First")
            elif event.key == pygame.K_4:
                self.handle_action("A*")
            elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                self.handle_action("faster")
            elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                self.handle_action("slower")
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                cell = self.grid_cell_at(event.pos)
                if cell is not None:
                    self.handle_grid_click(cell, pygame.key.get_mods())
                else:
                    for button in self.buttons:
                        if button.rect.collidepoint(event.pos):
                            self.handle_action(button.action)
                            break
            elif event.button == 3:
                # Right click clears a wall quickly.
                cell = self.grid_cell_at(event.pos)
                if cell in self.walls:
                    self.walls.remove(cell)
                    self.reset_search()

    def run(self):
        while self.running:
            dt = self.clock.tick(FPS) / 1000.0
            for event in pygame.event.get():
                self.handle_event(event)
            self.update_animation(dt)
            self.draw()
        pygame.quit()


def main():
    game = AIMazeLab()
    game.run()


if __name__ == "__main__":
    main()
